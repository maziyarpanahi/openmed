"""Portable export contracts for GLiNER span and GLiClass zero-shot profiles.

GLiNER span detection and GLiClass zero-shot classification are
label-conditioned: candidate labels are part of the model input, so their
exported graphs do not fit the ``(input_ids, attention_mask) -> logits``
token-classification converter contract. This module defines one versioned
contract per supported profile and enforces it on both sides of an export:

- the prompt and candidate-label encoding (marker tokens, label ordering, the
  word splitter, and how tokens map back to source code-point offsets);
- stable tensor names, dtypes, and shapes of the exported graph;
- declared bounds (labels, sequence length, words, span width, label length)
  and a deterministic activation-memory estimate;
- the decoder, threshold, overlap, and tie-breaking rules.

Requests outside the declared bounds are refused before inference with a
typed :class:`ZeroShotUnsupported` result carrying a stable, value-free
:class:`~openmed.eval.export_qualification.ExportReason`. Relation extraction
(``gliner-uni-encoder-token-relex``) has no portable profile and always fails
with ``unsupported_task``. Nothing here imports a model runtime; tokenizers
are duck-typed ``tokenizers.Tokenizer`` objects loaded from local files.

Exported artifacts reproduce the source model's outputs; they never broaden a
model's clinical claim and never trigger a clinical decision.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from openmed.eval.export_qualification import (
    QUANTIZATION_NONE,
    SHAPE_MODE_DYNAMIC,
    ExportIOContract,
    ExportQualificationError,
    ExportReason,
    RuntimeSupport,
)

ZERO_SHOT_CONTRACT_VERSION = 1
ZERO_SHOT_MANIFEST_FORMAT = "openmed-zero-shot-export"
ZERO_SHOT_MANIFEST_SCHEMA_VERSION = 1
ZERO_SHOT_MANIFEST_FILENAME = "openmed-zero-shot.json"

GLINER_SPAN_FAMILY = "gliner-uni-encoder-span"
GLICLASS_FAMILY = "gliclass-uni-encoder"
GLINER_RELEX_FAMILY = "gliner-uni-encoder-token-relex"
TASK_ZERO_SHOT_NER = "zero-shot-ner"
TASK_ZERO_SHOT_CLASSIFICATION = "zero-shot-sequence-classification"
TASK_ZERO_SHOT_RELATION_EXTRACTION = "zero-shot-relation-extraction"
GLINER_SPAN_PROFILE_NAME = "gliner-span-uni-encoder-v1"
GLICLASS_PROFILE_NAME = "gliclass-uni-encoder-v1"
ZERO_SHOT_ADAPTER_VERSION = 1

ONNX_RUNTIME = "onnx"
ONNX_CPU_PROVIDER = "CPUExecutionProvider"
ZERO_SHOT_ONNX_OPSET = 18
ZERO_SHOT_ONNX_SUPPORT = RuntimeSupport(
    runtime=ONNX_RUNTIME,
    precisions=("float32",),
    quantizations=(QUANTIZATION_NONE,),
    providers=(ONNX_CPU_PROVIDER,),
    shape_modes=(SHAPE_MODE_DYNAMIC,),
    target=f"opset>={ZERO_SHOT_ONNX_OPSET}",
)
QUALIFIED_BATCH_SIZE = 1
DEFAULT_THRESHOLD = 0.5
DEFAULT_MAX_LABELS = 32
HARD_MAX_LABELS = 64
DEFAULT_MAX_SEQUENCE_LENGTH = 512
DEFAULT_MAX_LABEL_CHARS = 128
DEFAULT_GLINER_MAX_WORDS = 384
HARD_MAX_SPAN_WIDTH = 32
FLOAT32_BYTES = 4
# Python's ``str`` indexes Unicode code points, so offsets from this splitter
# are code-point offsets into the caller's text. It matches the MLX runtime.
GLINER_WORD_PATTERN = r"\w+(?:[-_]\w+)*|\S"
_WORD_RE = re.compile(GLINER_WORD_PATTERN)
GLINER_WORD_SPLITTER = {
    "name": "gliner-regex-words",
    "version": 1,
    "pattern": GLINER_WORD_PATTERN,
    "flags": "unicode",
}
ZERO_SHOT_OFFSET_CONTRACTS: Mapping[str, Mapping[str, str]] = {
    TASK_ZERO_SHOT_NER: {
        "source": "word_splitter",
        "unit": "unicode_code_point",
        "interval": "half_open",
        "span_start": "first_word_start",
        "span_end": "last_word_end",
        "prompt_tokens": "no_offsets",
    },
    TASK_ZERO_SHOT_CLASSIFICATION: {
        "source": "not_applicable",
        "unit": "unicode_code_point",
        "interval": "half_open",
        "prompt_tokens": "no_offsets",
    },
}
# Canonical OpenMed labels mapped to the natural-language prompts used when the
# packaged synthetic PHI suite qualifies a GLiNER span export.
DEFAULT_GLINER_LABEL_PROMPTS: Mapping[str, str] = {
    "ACCOUNT_NUMBER": "account number",
    "AGE": "age",
    "CREDIT_CARD": "credit card number",
    "DATE": "date",
    "DATE_OF_BIRTH": "date of birth",
    "EMAIL": "email",
    "ID_NUM": "identifier",
    "IP_ADDRESS": "ip address",
    "LOCATION": "location",
    "ORGANIZATION": "organization",
    "PERSON": "person",
    "PHONE": "phone number",
    "SSN": "social security number",
    "STREET_ADDRESS": "street address",
    "URL": "url",
}


class ZeroShotRequestError(ExportQualificationError):
    """A zero-shot request outside the declared contract.

    Args:
        reason: The stable reason code.
        dimension: The contract dimension that failed, for example
            ``"labels"`` or ``"sequence_length"``.
        limit: The declared bound for *dimension*, when there is one. It is a
            property of the artifact, never of the request.
    """

    def __init__(
        self,
        reason: ExportReason | str,
        *,
        dimension: str,
        limit: int | None = None,
    ) -> None:
        super().__init__(reason)
        self.dimension = dimension
        self.limit = limit


@dataclass(frozen=True)
class ZeroShotUnsupported:
    """Typed refusal for a request the qualified artifact does not cover.

    Attributes:
        reason: Stable reason code.
        dimension: Contract dimension that failed.
        limit: Declared bound for that dimension, if any.
        status: Always ``"unsupported"``.
    """

    reason: ExportReason
    dimension: str
    limit: int | None = None
    status: str = "unsupported"

    @property
    def reason_code(self) -> str:
        """Return the reason as a plain string."""
        return self.reason.value

    @property
    def ok(self) -> bool:
        """Return ``False``; an unsupported request produced no output."""
        return False

    @classmethod
    def from_error(cls, error: ExportQualificationError) -> "ZeroShotUnsupported":
        """Build a typed refusal from a contract error."""
        return cls(
            reason=error.reason,
            dimension=str(getattr(error, "dimension", "request")),
            limit=getattr(error, "limit", None),
        )

    def raise_for_status(self) -> None:
        """Raise the equivalent :class:`ZeroShotRequestError`."""
        raise ZeroShotRequestError(
            self.reason, dimension=self.dimension, limit=self.limit
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free JSON form."""
        return {
            "status": self.status,
            "reason": self.reason.value,
            "dimension": self.dimension,
            "limit": self.limit,
        }


@dataclass(frozen=True)
class ZeroShotSpan:
    """One decoded span; offsets index the caller's text, which is not kept.

    Attributes:
        label: Caller-supplied candidate label.
        label_index: Position of *label* in the caller's label list.
        start: Inclusive start code-point offset.
        end: Exclusive end code-point offset.
        score: Sigmoid score in ``[0, 1]``.
    """

    label: str
    label_index: int
    start: int
    end: int
    score: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping without source text."""
        return {
            "label": self.label,
            "label_index": self.label_index,
            "start": self.start,
            "end": self.end,
            "score": self.score,
        }


@dataclass(frozen=True)
class ZeroShotSpanResult:
    """Spans decoded from one GLiNER span request."""

    spans: tuple[ZeroShotSpan, ...]
    status: str = "ok"

    @property
    def ok(self) -> bool:
        """Return ``True``; the request ran inside the qualified bounds."""
        return True


@dataclass(frozen=True)
class ZeroShotLabelScore:
    """One candidate label score from a GLiClass request."""

    label: str
    label_index: int
    score: float
    accepted: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping."""
        return {
            "label": self.label,
            "label_index": self.label_index,
            "score": self.score,
            "accepted": self.accepted,
        }


@dataclass(frozen=True)
class ZeroShotClassificationResult:
    """Ranked label scores from one GLiClass request.

    ``scores`` is ordered by descending score with ties broken by caller label
    order; ``accepted`` marks scores at or above the threshold.
    """

    scores: tuple[ZeroShotLabelScore, ...]
    status: str = "ok"

    @property
    def ok(self) -> bool:
        """Return ``True``; the request ran inside the qualified bounds."""
        return True

    @property
    def accepted(self) -> tuple[ZeroShotLabelScore, ...]:
        """Return the accepted labels in ranking order."""
        return tuple(item for item in self.scores if item.accepted)


@dataclass(frozen=True)
class ZeroShotBounds:
    """Declared, enforced bounds of one exported zero-shot artifact.

    Attributes:
        max_labels: Largest number of candidate labels per request.
        max_sequence_length: Largest token count, prompt included.
        max_label_chars: Largest code-point length of one candidate label.
        max_words: GLiNER only: largest number of source words.
        max_span_width: GLiNER only: largest span width in words.
    """

    max_labels: int
    max_sequence_length: int
    max_label_chars: int = DEFAULT_MAX_LABEL_CHARS
    max_words: int | None = None
    max_span_width: int | None = None

    def __post_init__(self) -> None:
        values = [self.max_labels, self.max_sequence_length, self.max_label_chars]
        values.extend(
            value
            for value in (self.max_words, self.max_span_width)
            if value is not None
        )
        if any(type(value) is not int or value <= 0 for value in values):
            raise ExportQualificationError(ExportReason.IO_CONTRACT_MISMATCH)

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest form of the bounds."""
        return {
            "max_labels": self.max_labels,
            "max_sequence_length": self.max_sequence_length,
            "max_label_chars": self.max_label_chars,
            "max_words": self.max_words,
            "max_span_width": self.max_span_width,
            "qualified_batch_size": QUALIFIED_BATCH_SIZE,
            "overflow": "reject",
            "dynamic": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ZeroShotBounds":
        """Rebuild bounds from :meth:`to_dict` output.

        Raises:
            ExportQualificationError: ``io_contract_mismatch`` for a malformed
                or unbounded payload.
        """

        try:
            return cls(
                max_labels=payload["max_labels"],
                max_sequence_length=payload["max_sequence_length"],
                max_label_chars=payload["max_label_chars"],
                max_words=payload.get("max_words"),
                max_span_width=payload.get("max_span_width"),
            )
        except (KeyError, TypeError):
            raise ExportQualificationError(ExportReason.IO_CONTRACT_MISMATCH) from None


@dataclass(frozen=True)
class ZeroShotPromptSpec:
    """Prompt markers and layout for one exported artifact.

    Attributes:
        marker_token: ``<<ENT>>`` for GLiNER, ``<<LABEL>>`` for GLiClass.
        separator_token: Marker that closes the label block.
        marker_token_id: Vocabulary id of *marker_token*.
        separator_token_id: Vocabulary id of *separator_token*.
        embed_marker_token: Whether a label is represented by its marker
            token (``True``) or by the token after it (``False``).
        prompt_first: GLiClass only: label block before the text.
        reserved_markers: Strings that must never appear in labels or text.
    """

    marker_token: str
    separator_token: str
    marker_token_id: int
    separator_token_id: int
    embed_marker_token: bool = True
    prompt_first: bool = True
    reserved_markers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest form of the prompt contract."""
        return {
            "marker_token": self.marker_token,
            "separator_token": self.separator_token,
            "marker_token_id": self.marker_token_id,
            "separator_token_id": self.separator_token_id,
            "embed_marker_token": self.embed_marker_token,
            "prompt_first": self.prompt_first,
            "reserved_markers": list(self.reserved_markers),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ZeroShotPromptSpec":
        """Rebuild a prompt spec from :meth:`to_dict` output."""
        try:
            return cls(
                marker_token=str(payload["marker_token"]),
                separator_token=str(payload["separator_token"]),
                marker_token_id=int(payload["marker_token_id"]),
                separator_token_id=int(payload["separator_token_id"]),
                embed_marker_token=bool(payload["embed_marker_token"]),
                prompt_first=bool(payload.get("prompt_first", True)),
                reserved_markers=tuple(payload.get("reserved_markers") or ()),
            )
        except (KeyError, TypeError, ValueError):
            raise ExportQualificationError(ExportReason.IO_CONTRACT_MISMATCH) from None


@dataclass(frozen=True)
class ZeroShotProfile:
    """One declared portable zero-shot export profile.

    Attributes:
        name: Versioned profile name recorded in manifests.
        family: OpenMed family the profile accepts.
        task: Task the artifact performs.
        input_names: Graph input names in feed order.
        output_names: Graph output names.
        decoder: Versioned decoder name.
        default_marker: Default label marker token.
        default_separator: Default separator token.
        reserved_markers: Markers rejected in labels and text.
    """

    name: str
    family: str
    task: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    decoder: str
    default_marker: str
    default_separator: str
    reserved_markers: tuple[str, ...]
    adapter_version: int = ZERO_SHOT_ADAPTER_VERSION

    def support_for(self, runtime: str) -> RuntimeSupport:
        """Return the declared support for *runtime*.

        Core ML and every runtime other than ONNX Runtime have no qualified
        zero-shot profile.

        Raises:
            ExportQualificationError: ``unsupported_runtime``.
        """

        if runtime != ONNX_RUNTIME:
            raise ExportQualificationError(ExportReason.UNSUPPORTED_RUNTIME)
        return ZERO_SHOT_ONNX_SUPPORT

    def decoder_contract(self, threshold: float) -> dict[str, Any]:
        """Return the decoder and threshold contract for manifests."""
        if self.task == TASK_ZERO_SHOT_NER:
            return {
                "name": self.decoder,
                "version": 1,
                "activation": "sigmoid",
                "threshold": threshold,
                "threshold_rule": "score >= threshold",
                "invalid_spans": "masked_to_-1e9",
                "overlap": {
                    "flat_ner": "greedy_non_overlapping_half_open",
                    "nested": "keep_all",
                },
                "tie_break": ["score_desc", "start", "end", "label_index"],
                "output_order": ["start", "end", "label_index"],
            }
        return {
            "name": self.decoder,
            "version": 1,
            "activation": "sigmoid",
            "threshold": threshold,
            "threshold_rule": "score >= threshold",
            "ranking": ["score_desc", "label_index"],
            "multi_label": True,
        }

    def tensor_contract(self, *, span_width: int | None = None) -> dict[str, Any]:
        """Return stable tensor names, dtypes, and symbolic shapes."""
        if self.task == TASK_ZERO_SHOT_NER:
            return {
                "inputs": [
                    _tensor("input_ids", "int64", ["batch", "sequence"]),
                    _tensor("attention_mask", "int64", ["batch", "sequence"]),
                    _tensor("words_mask", "int64", ["batch", "sequence"]),
                    _tensor("text_lengths", "int64", ["batch", 1]),
                    _tensor("span_idx", "int64", ["batch", "spans", 2]),
                    _tensor("span_mask", "bool", ["batch", "spans"]),
                ],
                "outputs": [
                    _tensor(
                        "logits",
                        "float32",
                        ["batch", "words", span_width or "span_width", "labels"],
                    )
                ],
                "spans": "words * span_width",
            }
        return {
            "inputs": [
                _tensor("input_ids", "int64", ["batch", "sequence"]),
                _tensor("attention_mask", "int64", ["batch", "sequence"]),
            ],
            "outputs": [_tensor("logits", "float32", ["batch", "labels"])],
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready profile description."""
        return {
            "name": self.name,
            "family": self.family,
            "task": self.task,
            "adapter_version": self.adapter_version,
            "contract_version": ZERO_SHOT_CONTRACT_VERSION,
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "decoder": self.decoder,
            "support": {ONNX_RUNTIME: ZERO_SHOT_ONNX_SUPPORT.to_dict()},
        }


GLINER_SPAN_PROFILE = ZeroShotProfile(
    name=GLINER_SPAN_PROFILE_NAME,
    family=GLINER_SPAN_FAMILY,
    task=TASK_ZERO_SHOT_NER,
    input_names=(
        "input_ids",
        "attention_mask",
        "words_mask",
        "text_lengths",
        "span_idx",
        "span_mask",
    ),
    output_names=("logits",),
    decoder="gliner-span-sigmoid-greedy",
    default_marker="<<ENT>>",
    default_separator="<<SEP>>",
    reserved_markers=("<<ENT>>", "<<SEP>>", "<<REL>>"),
)
GLICLASS_PROFILE = ZeroShotProfile(
    name=GLICLASS_PROFILE_NAME,
    family=GLICLASS_FAMILY,
    task=TASK_ZERO_SHOT_CLASSIFICATION,
    input_names=("input_ids", "attention_mask"),
    output_names=("logits",),
    decoder="gliclass-sigmoid-threshold",
    default_marker="<<LABEL>>",
    default_separator="<<SEP>>",
    reserved_markers=("<<LABEL>>", "<<SEP>>", "<<EXAMPLE>>"),
)
ZERO_SHOT_PROFILES: Mapping[str, ZeroShotProfile] = {
    GLINER_SPAN_FAMILY: GLINER_SPAN_PROFILE,
    GLICLASS_FAMILY: GLICLASS_PROFILE,
}
_ZERO_SHOT_TASKS = {
    TASK_ZERO_SHOT_NER: GLINER_SPAN_FAMILY,
    TASK_ZERO_SHOT_CLASSIFICATION: GLICLASS_FAMILY,
}
_RELATION_FAMILIES = frozenset({GLINER_RELEX_FAMILY})
_ZERO_SHOT_MODEL_TYPES = frozenset({"gliner", "gliclass", "gliner2"})
_REMOTE_CODE_KEYS = ("auto_map", "trust_remote_code", "custom_pipelines")


@dataclass(frozen=True)
class ZeroShotModelSpec:
    """Value-free facts about a validated zero-shot checkpoint configuration.

    Attributes:
        profile: The matched export profile.
        hidden_size: Head hidden size.
        encoder_hidden_size: Backbone hidden size.
        num_hidden_layers: Backbone layers.
        num_attention_heads: Backbone attention heads.
        intermediate_size: Backbone feed-forward size.
        max_position_embeddings: Backbone position limit.
        marker_token_index: Label marker token id from the configuration.
        text_token_index: GLiClass text marker id.
        example_token_index: GLiClass example marker id, if any.
        span_width: GLiNER ``max_width``.
        max_words: GLiNER ``max_len`` word bound, if configured.
        options: Remaining variant switches recorded in the manifest.
    """

    profile: ZeroShotProfile
    hidden_size: int
    encoder_hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int
    marker_token_index: int
    text_token_index: int | None = None
    example_token_index: int | None = None
    span_width: int | None = None
    max_words: int | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest form of the architecture facts."""
        return {
            "hidden_size": self.hidden_size,
            "encoder_hidden_size": self.encoder_hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "marker_token_index": self.marker_token_index,
            "text_token_index": self.text_token_index,
            "example_token_index": self.example_token_index,
            "span_width": self.span_width,
            "max_words": self.max_words,
            "options": dict(self.options),
        }


# ---------------------------------------------------------------------------
# Profile resolution and configuration validation
# ---------------------------------------------------------------------------


def resolve_zero_shot_profile(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any] | None = None,
) -> ZeroShotProfile:
    """Return the export profile for a converted zero-shot checkpoint.

    Args:
        config: The checkpoint ``config.json`` mapping.
        manifest: Optional ``openmed-mlx.json`` manifest.

    Raises:
        ExportQualificationError: ``unsupported_remote_code`` when the
            checkpoint needs repository code, ``unsupported_task`` for
            relation extraction or a task the family does not perform,
            ``unsupported_quantization`` for a quantized source, or
            ``unsupported_family`` for any other family.
    """

    if any(config.get(key) for key in _REMOTE_CODE_KEYS):
        raise ExportQualificationError(ExportReason.UNSUPPORTED_REMOTE_CODE)
    family = _normalize(
        (manifest or {}).get("family") or config.get("_mlx_family") or ""
    )
    task = str((manifest or {}).get("task") or config.get("_mlx_task") or "")
    if family in _RELATION_FAMILIES or task == TASK_ZERO_SHOT_RELATION_EXTRACTION:
        raise ExportQualificationError(ExportReason.UNSUPPORTED_TASK)
    profile = ZERO_SHOT_PROFILES.get(family)
    if profile is None:
        raise ExportQualificationError(ExportReason.UNSUPPORTED_FAMILY)
    if task and task != profile.task:
        raise ExportQualificationError(ExportReason.UNSUPPORTED_TASK)
    if config.get("_mlx_quantization") or (manifest or {}).get("quantization"):
        raise ExportQualificationError(ExportReason.UNSUPPORTED_QUANTIZATION)
    return profile


def validate_zero_shot_config(
    profile: ZeroShotProfile,
    config: Mapping[str, Any],
) -> ZeroShotModelSpec:
    """Reject configurations the portable graph does not reproduce.

    *config* should already carry the MLX family defaults (see
    :func:`openmed.mlx.models.normalize_model_config`), so the exported graph
    and the MLX reference read identical values.

    Raises:
        ExportQualificationError: ``unsupported_model_variant`` for any
            switch outside the profile (scorer, pooling, span mode, LSTM
            depth, convolution, activation, missing marker ids).
    """

    _require_variant(int(config.get("conv_kernel_size") or 0) == 0)
    _require_variant(str(config.get("hidden_act", "gelu")) == "gelu")
    backbone = _normalize(config.get("_mlx_model_type") or config.get("model_type"))
    _require_variant(backbone in {"deberta-v2", "deberta"})
    hidden = _positive(config.get("hidden_size"))
    encoder_hidden = _positive(config.get("encoder_hidden_size", hidden))
    layers = _positive(config.get("num_hidden_layers"))
    heads = _positive(config.get("num_attention_heads"))
    intermediate = _positive(config.get("intermediate_size"))
    positions = _positive(config.get("max_position_embeddings"))
    marker = _token_index(config.get("class_token_index"))

    if profile.task == TASK_ZERO_SHOT_NER:
        span_mode = config.get("span_mode", "markerV0")
        _require_variant(span_mode in {None, "markerV0", "marker_v0"})
        rnn_layers = int(config.get("num_rnn_layers") or 0)
        _require_variant(rnn_layers in {0, 1})
        _require_variant(hidden % 2 == 0 if rnn_layers else True)
        width = _positive(config.get("max_width", 12))
        _require_variant(width <= HARD_MAX_SPAN_WIDTH)
        max_len = config.get("max_len")
        return ZeroShotModelSpec(
            profile=profile,
            hidden_size=hidden,
            encoder_hidden_size=encoder_hidden,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            intermediate_size=intermediate,
            max_position_embeddings=positions,
            marker_token_index=marker,
            span_width=width,
            max_words=int(max_len) if isinstance(max_len, int) and max_len else None,
            options={
                "span_mode": "markerV0",
                "num_rnn_layers": rnn_layers,
                "embed_ent_token": bool(config.get("embed_ent_token", True)),
            },
        )

    _require_variant(config.get("architecture_type", "uni-encoder") == "uni-encoder")
    _require_variant(config.get("scorer_type", "mlp") == "mlp")
    for switch in ("extract_text_features", "use_lstm", "squeeze_layers"):
        _require_variant(not config.get(switch))
    # The OpenMed MLX reference pools every label over its marker span until
    # the next marker or the text marker whatever this (normalized) field
    # says; see reject_raw_first_token_pooling for checkpoints that declare
    # first-token pooling explicitly.
    _require_variant(config.get("class_token_pooling") in {None, "first", "average"})
    pooling = config.get("pooling_strategy", "first")
    _require_variant(pooling in {"first", "mean"})
    text_marker = _token_index(config.get("text_token_index"))
    example = config.get("example_token_index")
    return ZeroShotModelSpec(
        profile=profile,
        hidden_size=hidden,
        encoder_hidden_size=encoder_hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        intermediate_size=intermediate,
        max_position_embeddings=positions,
        marker_token_index=marker,
        text_token_index=text_marker,
        example_token_index=_token_index(example) if example is not None else None,
        options={
            "pooling_strategy": pooling,
            "class_pooling": "span_mean_until_next_marker",
            "embed_class_token": bool(config.get("embed_class_token", True)),
            "use_segment_embeddings": bool(config.get("use_segment_embeddings")),
            "normalize_features": bool(config.get("normalize_features")),
        },
    )


def reject_raw_first_token_pooling(raw_config: Mapping[str, Any]) -> None:
    """Reject GLiClass checkpoints that explicitly declare first-token pooling.

    The MLX family defaults ``class_token_pooling`` to ``"first"`` while the
    OpenMed reference always span-averages labels, so only the *raw*
    ``config.json`` tells whether a checkpoint was trained with first-token
    pooling.

    Raises:
        ExportQualificationError: ``unsupported_model_variant``.
    """

    if raw_config.get("class_token_pooling") == "first":
        raise ExportQualificationError(ExportReason.UNSUPPORTED_MODEL_VARIANT)


def resolve_prompt_spec(
    profile: ZeroShotProfile,
    config: Mapping[str, Any],
    spec: ZeroShotModelSpec,
    tokenizer: Any,
    *,
    prompt_spec: Mapping[str, Any] | None = None,
) -> ZeroShotPromptSpec:
    """Return the prompt contract and check it against the tokenizer.

    Args:
        profile: The matched export profile.
        config: Normalized checkpoint configuration.
        spec: Validated architecture facts.
        tokenizer: ``tokenizers.Tokenizer``-like object.
        prompt_spec: The MLX manifest ``prompt_spec``; defaults to the
            configuration's ``_mlx_prompt_spec``.

    Raises:
        ExportQualificationError: ``unsupported_model_variant`` when a marker
            is not a single vocabulary token or its id disagrees with the
            configuration.
    """

    prompt = dict(prompt_spec or config.get("_mlx_prompt_spec") or {})
    if profile.task == TASK_ZERO_SHOT_NER:
        marker = str(prompt.get("entity_token", profile.default_marker))
        embed = bool(spec.options.get("embed_ent_token", True))
        prompt_first = True
    else:
        marker = str(prompt.get("label_token", profile.default_marker))
        embed = bool(spec.options.get("embed_class_token", True))
        prompt_first = bool(prompt.get("prompt_first", True))
    separator = str(prompt.get("separator_token", profile.default_separator))
    marker_id = _vocab_id(tokenizer, marker)
    separator_id = _vocab_id(tokenizer, separator)
    _require_variant(marker_id == spec.marker_token_index)
    if profile.task == TASK_ZERO_SHOT_CLASSIFICATION:
        _require_variant(separator_id == spec.text_token_index)
    reserved = tuple(dict.fromkeys((*profile.reserved_markers, marker, separator)))
    return ZeroShotPromptSpec(
        marker_token=marker,
        separator_token=separator,
        marker_token_id=marker_id,
        separator_token_id=separator_id,
        embed_marker_token=embed,
        prompt_first=prompt_first,
        reserved_markers=reserved,
    )


def default_bounds(
    spec: ZeroShotModelSpec,
    *,
    max_labels: int | None = None,
    max_sequence_length: int | None = None,
    max_words: int | None = None,
    max_span_width: int | None = None,
    max_label_chars: int | None = None,
) -> ZeroShotBounds:
    """Return declared bounds, rejecting requests the model cannot honor.

    Raises:
        ExportQualificationError: ``unsupported_label_count``,
            ``unsupported_sequence_length``, ``unsupported_word_count``, or
            ``unsupported_span_width`` when a requested bound exceeds what
            the checkpoint or the profile supports.
    """

    labels = DEFAULT_MAX_LABELS if max_labels is None else max_labels
    if type(labels) is not int or not 0 < labels <= HARD_MAX_LABELS:
        raise ZeroShotRequestError(
            ExportReason.UNSUPPORTED_LABEL_COUNT,
            dimension="labels",
            limit=HARD_MAX_LABELS,
        )
    position_limit = spec.max_position_embeddings
    sequence = (
        min(DEFAULT_MAX_SEQUENCE_LENGTH, position_limit)
        if max_sequence_length is None
        else max_sequence_length
    )
    if type(sequence) is not int or not 0 < sequence <= position_limit:
        raise ZeroShotRequestError(
            ExportReason.UNSUPPORTED_SEQUENCE_LENGTH,
            dimension="sequence_length",
            limit=position_limit,
        )
    chars = DEFAULT_MAX_LABEL_CHARS if max_label_chars is None else max_label_chars
    if spec.profile.task != TASK_ZERO_SHOT_NER:
        if max_words is not None or max_span_width is not None:
            raise ExportQualificationError(ExportReason.IO_CONTRACT_MISMATCH)
        return ZeroShotBounds(
            max_labels=labels, max_sequence_length=sequence, max_label_chars=chars
        )
    word_limit = min(spec.max_words or DEFAULT_GLINER_MAX_WORDS, sequence)
    words = word_limit if max_words is None else max_words
    if type(words) is not int or not 0 < words <= word_limit:
        raise ZeroShotRequestError(
            ExportReason.UNSUPPORTED_WORD_COUNT, dimension="words", limit=word_limit
        )
    graph_width = int(spec.span_width or 1)
    width = graph_width if max_span_width is None else max_span_width
    if type(width) is not int or not 0 < width <= graph_width:
        raise ZeroShotRequestError(
            ExportReason.UNSUPPORTED_SPAN_WIDTH,
            dimension="span_width",
            limit=graph_width,
        )
    return ZeroShotBounds(
        max_labels=labels,
        max_sequence_length=sequence,
        max_label_chars=chars,
        max_words=words,
        max_span_width=width,
    )


def reject_zero_shot_token_classification(
    config: Any,
    *,
    runtime: str,
) -> None:
    """Refuse a zero-shot checkpoint in a token-classification converter.

    GLiNER and GLiClass checkpoints often declare a DeBERTa ``model_type``,
    so without this guard a token-classification exporter could load their
    backbone, attach a random head, and silently change the task.

    Args:
        config: Hugging Face configuration object or mapping.
        runtime: ``"coreml"`` or ``"onnx"``.

    Raises:
        ExportQualificationError: ``unsupported_task`` for relation
            extraction and for the ONNX token-classification converter (the
            zero-shot exporter is :mod:`openmed.onnx.zero_shot_export`), or
            ``unsupported_runtime`` for Core ML, which has no qualified
            zero-shot profile.
    """

    family = _normalize(_config_get(config, "_mlx_family") or "")
    task = str(_config_get(config, "_mlx_task") or "")
    model_type = _normalize(_config_get(config, "model_type") or "")
    zero_shot = (
        family in ZERO_SHOT_PROFILES
        or family in _RELATION_FAMILIES
        or task.startswith("zero-shot")
        or model_type in _ZERO_SHOT_MODEL_TYPES
    )
    if not zero_shot:
        return
    if family in _RELATION_FAMILIES or task == TASK_ZERO_SHOT_RELATION_EXTRACTION:
        raise ExportQualificationError(ExportReason.UNSUPPORTED_TASK)
    if runtime == "coreml":
        raise ExportQualificationError(
            ExportReason.UNSUPPORTED_RUNTIME,
            "unsupported_runtime: Core ML has no qualified GLiNER or GLiClass profile",
        )
    raise ExportQualificationError(
        ExportReason.UNSUPPORTED_TASK,
        "unsupported_task: label-conditioned checkpoints use "
        "openmed.onnx.zero_shot_export",
    )


# ---------------------------------------------------------------------------
# Request checks and encoding
# ---------------------------------------------------------------------------


def split_words(text: str) -> list[tuple[str, int, int]]:
    """Split *text* into ``(word, start, end)`` code-point spans.

    This is the GLiNER word splitter shared with the MLX runtime.
    """

    return [
        (match.group(0), match.start(), match.end())
        for match in _WORD_RE.finditer(text)
    ]


def check_labels(
    labels: Sequence[str],
    *,
    bounds: ZeroShotBounds,
    prompt: ZeroShotPromptSpec,
) -> tuple[str, ...]:
    """Validate candidate labels and return them in caller order.

    Raises:
        ZeroShotRequestError: ``unsupported_label_count`` for an empty or
            oversized list; ``unsupported_label`` for a non-string, blank,
            oversized, duplicated, or control-character label; or
            ``unsupported_control_marker`` for a label containing a marker.
    """

    if isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence):
        raise ZeroShotRequestError(ExportReason.UNSUPPORTED_LABEL, dimension="labels")
    if not 0 < len(labels) <= bounds.max_labels:
        raise ZeroShotRequestError(
            ExportReason.UNSUPPORTED_LABEL_COUNT,
            dimension="labels",
            limit=bounds.max_labels,
        )
    checked: list[str] = []
    for label in labels:
        if not isinstance(label, str) or not label.strip():
            raise ZeroShotRequestError(
                ExportReason.UNSUPPORTED_LABEL, dimension="label"
            )
        if len(label) > bounds.max_label_chars:
            raise ZeroShotRequestError(
                ExportReason.UNSUPPORTED_LABEL,
                dimension="label_chars",
                limit=bounds.max_label_chars,
            )
        if any(_is_control(char) for char in label):
            raise ZeroShotRequestError(
                ExportReason.UNSUPPORTED_LABEL, dimension="label"
            )
        if any(marker in label for marker in prompt.reserved_markers):
            raise ZeroShotRequestError(
                ExportReason.UNSUPPORTED_CONTROL_MARKER, dimension="label"
            )
        checked.append(label)
    if len(set(checked)) != len(checked):
        raise ZeroShotRequestError(ExportReason.UNSUPPORTED_LABEL, dimension="labels")
    return tuple(checked)


@dataclass(frozen=True)
class ZeroShotEncoding:
    """One encoded request; held in memory only and never serialized.

    Attributes:
        feed: Graph inputs keyed by the profile's input names.
        num_labels: Number of candidate labels in the prompt.
        token_count: Token count including the prompt.
        word_offsets: GLiNER only: code-point offsets of each source word.
        span_width: GLiNER only: requested span width.
    """

    feed: Mapping[str, Any]
    num_labels: int
    token_count: int
    word_offsets: tuple[tuple[int, int], ...] = ()
    span_width: int | None = None


def encode_gliner_span(
    tokenizer: Any,
    text: str,
    labels: Sequence[str],
    *,
    prompt: ZeroShotPromptSpec,
    bounds: ZeroShotBounds,
    graph_span_width: int,
    span_width: int | None = None,
) -> ZeroShotEncoding:
    """Encode one GLiNER span request under the profile contract.

    The prompt is ``[marker, label]`` per label in caller order, then the
    separator, then the source words. ``words_mask`` holds the 1-based word
    index on the first token of each source word. Candidate spans enumerate
    every start word and every width below *graph_span_width*; spans wider
    than *span_width* or past the last word are masked.

    Args:
        tokenizer: ``tokenizers.Tokenizer``-like object with
            ``encode(words, is_pretokenized=True)``.
        text: Source text; it is never logged or stored.
        labels: Candidate labels, already validated by :func:`check_labels`.
        prompt: Prompt contract of the artifact.
        bounds: Declared bounds of the artifact.
        graph_span_width: The exported graph's ``max_width``.
        span_width: Requested span width, at most ``bounds.max_span_width``.

    Raises:
        ZeroShotRequestError: For word count, span width, sequence length,
            control-marker, or offset-reconstruction violations.
    """

    import numpy as np

    width_limit = int(bounds.max_span_width or graph_span_width)
    width = width_limit if span_width is None else span_width
    if type(width) is not int or not 0 < width <= width_limit:
        raise ZeroShotRequestError(
            ExportReason.UNSUPPORTED_SPAN_WIDTH,
            dimension="span_width",
            limit=width_limit,
        )
    words = split_words(text)
    if bounds.max_words is not None and len(words) > bounds.max_words:
        raise ZeroShotRequestError(
            ExportReason.UNSUPPORTED_WORD_COUNT,
            dimension="words",
            limit=bounds.max_words,
        )
    prompt_words: list[str] = []
    for label in labels:
        prompt_words.extend([prompt.marker_token, label])
    prompt_words.append(prompt.separator_token)
    encoding = tokenizer.encode(
        prompt_words + [word for word, _, _ in words], is_pretokenized=True
    )
    input_ids = [int(value) for value in encoding.ids]
    if len(input_ids) > bounds.max_sequence_length:
        raise ZeroShotRequestError(
            ExportReason.UNSUPPORTED_SEQUENCE_LENGTH,
            dimension="sequence_length",
            limit=bounds.max_sequence_length,
        )
    if input_ids.count(prompt.marker_token_id) != len(labels):
        raise ZeroShotRequestError(
            ExportReason.UNSUPPORTED_CONTROL_MARKER, dimension="prompt"
        )
    offset = len(prompt_words)
    words_mask: list[int] = []
    seen: set[int] = set()
    previous: int | None = None
    for word_id in encoding.word_ids:
        if word_id is None or word_id < offset or word_id == previous:
            words_mask.append(0)
        else:
            words_mask.append(word_id - offset + 1)
            seen.add(word_id - offset)
        previous = word_id
    if seen != set(range(len(words))):
        raise ZeroShotRequestError(
            ExportReason.MISSING_OFFSET_MAPPING, dimension="words"
        )
    num_words = len(words)
    span_rows: list[list[int]] = []
    span_valid: list[bool] = []
    for start in range(num_words):
        for step in range(graph_span_width):
            span_rows.append([start, start + step])
            span_valid.append(step < width and start + step < num_words)
    attention = [int(value) for value in getattr(encoding, "attention_mask", [])]
    if len(attention) != len(input_ids):
        attention = [1] * len(input_ids)
    feed = {
        "input_ids": np.asarray([input_ids], dtype=np.int64),
        "attention_mask": np.asarray([attention], dtype=np.int64),
        "words_mask": np.asarray([words_mask], dtype=np.int64),
        "text_lengths": np.asarray([[num_words]], dtype=np.int64),
        "span_idx": np.asarray(span_rows, dtype=np.int64).reshape(1, -1, 2),
        "span_mask": np.asarray([span_valid], dtype=bool).reshape(1, -1),
    }
    return ZeroShotEncoding(
        feed=feed,
        num_labels=len(labels),
        token_count=len(input_ids),
        word_offsets=tuple((start, end) for _, start, end in words),
        span_width=width,
    )


def encode_gliclass(
    tokenizer: Any,
    text: str,
    labels: Sequence[str],
    *,
    prompt: ZeroShotPromptSpec,
    bounds: ZeroShotBounds,
    instruction: str | None = None,
) -> ZeroShotEncoding:
    """Encode one GLiClass request under the profile contract.

    The layout is ``marker + label`` per label in caller order, then the
    separator and the optional instruction, followed (``prompt_first``) or
    preceded by the text. Few-shot examples are not part of this profile.

    Raises:
        ZeroShotRequestError: ``unsupported_control_marker`` when the text or
            instruction contains a reserved marker or the encoded markers do
            not match the labels; ``unsupported_sequence_length`` when the
            encoding exceeds the bound (inputs are never truncated).
    """

    import numpy as np

    for value in (text, instruction or ""):
        if any(marker in value for marker in prompt.reserved_markers):
            raise ZeroShotRequestError(
                ExportReason.UNSUPPORTED_CONTROL_MARKER, dimension="text"
            )
    head = "".join(f"{prompt.marker_token}{label}" for label in labels)
    head += prompt.separator_token
    if instruction:
        head += instruction
    rendered = head + text if prompt.prompt_first else text + head
    encoding = tokenizer.encode(rendered)
    input_ids = [int(value) for value in encoding.ids]
    if len(input_ids) > bounds.max_sequence_length:
        raise ZeroShotRequestError(
            ExportReason.UNSUPPORTED_SEQUENCE_LENGTH,
            dimension="sequence_length",
            limit=bounds.max_sequence_length,
        )
    if (
        input_ids.count(prompt.marker_token_id) != len(labels)
        or input_ids.count(prompt.separator_token_id) != 1
    ):
        raise ZeroShotRequestError(
            ExportReason.UNSUPPORTED_CONTROL_MARKER, dimension="prompt"
        )
    attention = [int(value) for value in getattr(encoding, "attention_mask", [])]
    if len(attention) != len(input_ids):
        attention = [1] * len(input_ids)
    return ZeroShotEncoding(
        feed={
            "input_ids": np.asarray([input_ids], dtype=np.int64),
            "attention_mask": np.asarray([attention], dtype=np.int64),
        },
        num_labels=len(labels),
        token_count=len(input_ids),
    )


def estimate_activation_bytes(
    spec: ZeroShotModelSpec,
    *,
    sequence_length: int,
    num_labels: int,
    num_words: int = 0,
    span_width: int | None = None,
) -> int:
    """Return a deterministic float32 activation estimate for one request.

    The estimate covers the encoder hidden states, the disentangled attention
    score tensors, the feed-forward block, and the head activations. It is an
    upper-bound planning figure for memory budgets, not a measured peak.
    """

    seq = int(sequence_length)
    encoder = (
        seq * spec.encoder_hidden_size * 4
        + 3 * spec.num_attention_heads * seq * seq
        + seq * spec.intermediate_size
    )
    if spec.profile.task == TASK_ZERO_SHOT_NER:
        width = int(span_width or spec.span_width or 1)
        spans = int(num_words) * width
        head = (
            num_words * spec.hidden_size * 3
            + spans * spec.hidden_size * 6
            + spans * num_labels
            + num_labels * spec.hidden_size * 5
        )
    else:
        head = num_labels * (spec.encoder_hidden_size * 3 + 256 + 128)
    return int((encoder + head) * FLOAT32_BYTES)


def check_memory_budget(
    spec: ZeroShotModelSpec,
    *,
    budget_bytes: int | None,
    sequence_length: int,
    num_labels: int,
    num_words: int = 0,
    span_width: int | None = None,
) -> int:
    """Return the activation estimate, refusing one above *budget_bytes*.

    Raises:
        ZeroShotRequestError: ``unsupported_memory_budget``.
    """

    estimate = estimate_activation_bytes(
        spec,
        sequence_length=sequence_length,
        num_labels=num_labels,
        num_words=num_words,
        span_width=span_width,
    )
    if budget_bytes is not None and estimate > budget_bytes:
        raise ZeroShotRequestError(
            ExportReason.UNSUPPORTED_MEMORY_BUDGET,
            dimension="activation_bytes",
            limit=int(budget_bytes),
        )
    return estimate


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def decode_gliner_spans(
    logits: Any,
    *,
    word_offsets: Sequence[tuple[int, int]],
    labels: Sequence[str],
    threshold: float = DEFAULT_THRESHOLD,
    flat_ner: bool = True,
    span_width: int | None = None,
) -> ZeroShotSpanResult:
    """Decode ``[words, span_width, labels]`` span logits into spans.

    Scores are ``sigmoid(logit)``; spans at or above *threshold* are kept.
    With *flat_ner* a greedy pass keeps the highest-scoring non-overlapping
    spans (half-open code-point intervals), breaking ties by start, end, and
    caller label order. Results are ordered by start, end, label order.

    Raises:
        ExportQualificationError: ``io_contract_mismatch`` for logits whose
            shape does not cover the words and labels.
    """

    import numpy as np

    values = np.asarray(logits, dtype=np.float64)
    if values.ndim == 4:
        if values.shape[0] != 1:
            raise ExportQualificationError(ExportReason.IO_CONTRACT_MISMATCH)
        values = values[0]
    num_words = len(word_offsets)
    num_labels = len(labels)
    if values.ndim != 3 or (num_words and values.shape[0] != num_words):
        raise ExportQualificationError(ExportReason.IO_CONTRACT_MISMATCH)
    if num_words == 0:
        return ZeroShotSpanResult(spans=())
    if values.shape[2] < num_labels:
        raise ExportQualificationError(ExportReason.IO_CONTRACT_MISMATCH)
    width = values.shape[1] if span_width is None else min(span_width, values.shape[1])
    scores = _sigmoid(values[:, :width, :num_labels])
    candidates: list[ZeroShotSpan] = []
    for start, step, label_index in zip(*np.nonzero(scores >= threshold)):
        end_word = int(start) + int(step)
        if end_word >= num_words:
            continue
        candidates.append(
            ZeroShotSpan(
                label=str(labels[int(label_index)]),
                label_index=int(label_index),
                start=int(word_offsets[int(start)][0]),
                end=int(word_offsets[end_word][1]),
                score=float(scores[start, step, label_index]),
            )
        )
    if flat_ner:
        selected: list[ZeroShotSpan] = []
        for span in sorted(
            candidates,
            key=lambda item: (-item.score, item.start, item.end, item.label_index),
        ):
            if all(
                span.end <= other.start or span.start >= other.end for other in selected
            ):
                selected.append(span)
        candidates = selected
    ordered = sorted(
        candidates, key=lambda item: (item.start, item.end, item.label_index)
    )
    return ZeroShotSpanResult(spans=tuple(ordered))


def decode_gliclass(
    logits: Any,
    *,
    labels: Sequence[str],
    threshold: float = DEFAULT_THRESHOLD,
) -> ZeroShotClassificationResult:
    """Decode ``[labels]`` logits into ranked, thresholded label scores.

    Raises:
        ExportQualificationError: ``io_contract_mismatch`` for logits that do
            not cover every label.
    """

    import numpy as np

    values = np.asarray(logits, dtype=np.float64)
    if values.ndim == 2:
        if values.shape[0] != 1:
            raise ExportQualificationError(ExportReason.IO_CONTRACT_MISMATCH)
        values = values[0]
    if values.ndim != 1 or values.shape[0] < len(labels):
        raise ExportQualificationError(ExportReason.IO_CONTRACT_MISMATCH)
    scores = _sigmoid(values[: len(labels)])
    order = sorted(range(len(labels)), key=lambda index: (-scores[index], index))
    return ZeroShotClassificationResult(
        scores=tuple(
            ZeroShotLabelScore(
                label=str(labels[index]),
                label_index=index,
                score=float(scores[index]),
                accepted=bool(scores[index] >= threshold),
            )
            for index in order
        )
    )


# ---------------------------------------------------------------------------
# Harness integration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZeroShotIOContract(ExportIOContract):
    """:class:`ExportIOContract` with the zero-shot tensor shapes and bounds.

    The base fields keep :func:`check_qualified_request` working (input
    names, maximum sequence length, dynamic sequence axis); the extra fields
    record the exact tensor specs and every declared bound.
    """

    tensors: Mapping[str, Any] = field(default_factory=dict)
    bounds: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest form, with exact tensor shapes."""
        payload = super().to_dict()
        if self.tensors:
            payload["inputs"] = list(self.tensors.get("inputs") or [])
            payload["outputs"] = list(self.tensors.get("outputs") or [])
            extra = {
                key: value
                for key, value in self.tensors.items()
                if key not in {"inputs", "outputs"}
            }
            payload.update(extra)
        payload["bounds"] = dict(self.bounds)
        return payload


def build_io_contract(
    profile: ZeroShotProfile,
    bounds: ZeroShotBounds,
    *,
    graph_span_width: int | None = None,
) -> ZeroShotIOContract:
    """Return the harness tensor contract for one exported artifact."""
    tensors = profile.tensor_contract(span_width=graph_span_width)
    for spec in tensors["inputs"]:
        spec["shape"] = [
            QUALIFIED_BATCH_SIZE if dim == "batch" else dim for dim in spec["shape"]
        ]
    for spec in tensors["outputs"]:
        spec["shape"] = [
            QUALIFIED_BATCH_SIZE if dim == "batch" else dim for dim in spec["shape"]
        ]
    return ZeroShotIOContract(
        input_names=profile.input_names,
        output_names=profile.output_names,
        sequence_buckets=(),
        max_sequence_length=bounds.max_sequence_length,
        shape_mode=SHAPE_MODE_DYNAMIC,
        batch_size=QUALIFIED_BATCH_SIZE,
        input_dtype="int64",
        tensors=tensors,
        bounds=bounds.to_dict(),
    )


def label_schema_identity(
    profile: ZeroShotProfile,
    prompt: ZeroShotPromptSpec,
    bounds: ZeroShotBounds,
) -> dict[str, Any]:
    """Return the label-schema identity recorded as the report's label map.

    Zero-shot artifacts have no fixed label map: candidate labels are
    supplied per request. The identity therefore digests the label contract
    (markers, ordering, bounds) instead of any label text.
    """

    import hashlib
    import json

    payload = {
        "profile": profile.name,
        "ordering": "caller_order",
        "duplicates": "reject",
        "prompt": prompt.to_dict(),
        "max_labels": bounds.max_labels,
        "max_label_chars": bounds.max_label_chars,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "kind": "caller_supplied_labels",
        "num_labels": bounds.max_labels,
        "ordering": "caller_order",
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _tensor(name: str, dtype: str, shape: list[Any]) -> dict[str, Any]:
    return {"name": name, "dtype": dtype, "shape": list(shape)}


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _config_get(config: Any, key: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(key)
    return getattr(config, key, None)


def _require_variant(condition: bool) -> None:
    if not condition:
        raise ExportQualificationError(ExportReason.UNSUPPORTED_MODEL_VARIANT)


def _positive(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExportQualificationError(ExportReason.UNSUPPORTED_MODEL_VARIANT)
    return value


def _token_index(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExportQualificationError(ExportReason.UNSUPPORTED_MODEL_VARIANT)
    return value


def _vocab_id(tokenizer: Any, token: str) -> int:
    token_to_id = getattr(tokenizer, "token_to_id", None)
    value = token_to_id(token) if callable(token_to_id) else None
    if value is None:
        raise ExportQualificationError(ExportReason.UNSUPPORTED_MODEL_VARIANT)
    return int(value)


def _sigmoid(values: Any) -> Any:
    import numpy as np

    # Masked spans carry -1e9 logits; exp overflows to inf and the score is 0.
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-values))


def _is_control(char: str) -> bool:
    category = unicodedata.category(char)
    return category in {"Cc", "Cs"} or (category == "Cf" and char != "‍")


__all__ = [
    "DEFAULT_GLINER_LABEL_PROMPTS",
    "DEFAULT_THRESHOLD",
    "GLICLASS_FAMILY",
    "GLICLASS_PROFILE",
    "GLINER_RELEX_FAMILY",
    "GLINER_SPAN_FAMILY",
    "GLINER_SPAN_PROFILE",
    "GLINER_WORD_PATTERN",
    "GLINER_WORD_SPLITTER",
    "TASK_ZERO_SHOT_CLASSIFICATION",
    "TASK_ZERO_SHOT_NER",
    "TASK_ZERO_SHOT_RELATION_EXTRACTION",
    "ZERO_SHOT_CONTRACT_VERSION",
    "ZERO_SHOT_MANIFEST_FILENAME",
    "ZERO_SHOT_MANIFEST_FORMAT",
    "ZERO_SHOT_OFFSET_CONTRACTS",
    "ZERO_SHOT_ONNX_SUPPORT",
    "ZERO_SHOT_PROFILES",
    "ZeroShotBounds",
    "ZeroShotClassificationResult",
    "ZeroShotEncoding",
    "ZeroShotIOContract",
    "ZeroShotLabelScore",
    "ZeroShotModelSpec",
    "ZeroShotProfile",
    "ZeroShotPromptSpec",
    "ZeroShotRequestError",
    "ZeroShotSpan",
    "ZeroShotSpanResult",
    "ZeroShotUnsupported",
    "build_io_contract",
    "check_labels",
    "check_memory_budget",
    "decode_gliclass",
    "decode_gliner_spans",
    "default_bounds",
    "encode_gliclass",
    "encode_gliner_span",
    "estimate_activation_bytes",
    "label_schema_identity",
    "reject_raw_first_token_pooling",
    "reject_zero_shot_token_classification",
    "resolve_prompt_spec",
    "resolve_zero_shot_profile",
    "split_words",
    "validate_zero_shot_config",
]
