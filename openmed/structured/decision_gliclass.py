"""GLiClass v4 fixed-option decision backend over torch and ONNX Runtime.

``knowledgator/gliclass-v4-edge`` is an Apache-2.0 DeBERTa-v3 uni-encoder that
scores every caller-declared label in one forward pass. It reads an optional
instruction prompt placed before the text, optional few-shot example tokens,
and applies a learned calibrator to its logits. This module maps the
fixed-option decision contract in :mod:`openmed.structured.decision` onto that
model:

* ``fixed_choice``, ``boolean_choice`` and ``ordered_preference`` score the
  request options as one single-label question (softmax over the options);
* ``multi_label`` scores the options as independent labels (sigmoid per
  option);
* ``scalar_score`` scores configured, ordered score levels as one single-label
  question and returns the expected level scaled to ``[0, 1]``.

Calibrated scores drive the decision. The uncalibrated ``initial_score`` values
are kept in ``DecisionResult.extensions["gliclass"]`` by
:func:`decide_with_gliclass` whenever the runtime exposes them: the torch path
does, while the ``onnx/model_q8.onnx`` graph emits calibrated logits only.

The torch path needs ``pip install openmed[gliclass]``; ``gliclass`` requires
``transformers>=5`` and scikit-learn, so it is not part of the base install.
The ONNX path needs ``pip install openmed[onnx-runtime]``. Model files are
downloaded once at a pinned revision and scoring never touches the network.
Request text, instructions, few-shot examples and option values are never
logged or copied into errors or metadata. Results remain assistive review
artifacts and never authorize an autonomous clinical action.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import numbers
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

from openmed.ner.exceptions import MissingDependencyError

from .decision import (
    MAX_OPTION_BYTES,
    MAX_OPTION_CHARS,
    MAX_OPTIONS,
    DecisionAccessPolicy,
    DecisionBackendIdentity,
    DecisionBackendOutput,
    DecisionCalibrationProfile,
    DecisionConflictError,
    DecisionError,
    DecisionMode,
    DecisionRequest,
    DecisionResult,
    DecisionState,
    DecisionUnsupportedError,
    EncoderDecisionBackend,
    decide,
)

GLICLASS_V4_EDGE_MODEL_ID: Final = "knowledgator/gliclass-v4-edge"
GLICLASS_V4_EDGE_REVISION: Final = "885af1af8a07edf4ed34df83ca0d995ed8201283"
GLICLASS_V4_EDGE_LICENSE: Final = "Apache-2.0"
GLICLASS_ONNX_Q8_FILE: Final = "onnx/model_q8.onnx"
GLICLASS_ONNX_TOKENIZER_FILE: Final = "onnx/tokenizer.json"
GLICLASS_DEFAULT_MAX_LENGTH: Final = 2048
GLICLASS_DEFAULT_SCORE_LEVELS: Final = (
    "very low",
    "low",
    "moderate",
    "high",
    "very high",
)
GLICLASS_TORCH_INSTALL_HINT: Final = (
    "Install with `pip install openmed[gliclass]` to enable the GLiClass torch "
    "decision backend (gliclass requires transformers>=5 and scikit-learn)."
)
GLICLASS_ONNX_INSTALL_HINT: Final = (
    "Install with `pip install openmed[onnx-runtime]` to enable the GLiClass "
    "ONNX decision backend."
)
GLICLASS_METADATA_SCHEMA: Final = "openmed.gliclass.decision.v1"
MAX_PROMPT_CHARS: Final = 4096
MAX_PROMPT_BYTES: Final = 16384
MAX_EXAMPLES: Final = 8
MAX_EXAMPLE_CHARS: Final = 2048
MAX_EXAMPLE_BYTES: Final = 8192
MAX_SCORE_LEVELS: Final = 16

_LABEL_TOKEN: Final = "<<LABEL>>"
_SEP_TOKEN: Final = "<<SEP>>"
_EXAMPLE_TOKEN: Final = "<<EXAMPLE>>"
_CONTROL_MARKERS: Final = (_LABEL_TOKEN, _SEP_TOKEN, _EXAMPLE_TOKEN)
_ONNX_INPUTS: Final = frozenset({"input_ids", "attention_mask"})
_ONNX_OUTPUT: Final = "logits"
_TORCH_CHECKPOINT_FILES: Final = (
    "config.json",
    "*.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "spm.model",
)


class GLiClassClassificationType(str, Enum):
    """GLiClass scoring modes; values match ``classification_type``."""

    SINGLE_LABEL = "single-label"
    MULTI_LABEL = "multi-label"


@dataclass(frozen=True, slots=True)
class GLiClassExample:
    """One synthetic few-shot example rendered as GLiClass example tokens.

    Attributes:
        text: Example input. It is sent to the model and never logged.
        labels: Labels that apply to the example, in display order.
    """

    text: str = field(repr=False)
    labels: tuple[str, ...] = field(repr=False)

    def __post_init__(self) -> None:
        text = _checked_text(
            self.text,
            "example text",
            maximum_chars=MAX_EXAMPLE_CHARS,
            maximum_bytes=MAX_EXAMPLE_BYTES,
            multiline=True,
        )
        if isinstance(self.labels, (str, bytes)) or not isinstance(
            self.labels, Sequence
        ):
            raise TypeError("example labels must be a sequence of strings")
        labels = tuple(_checked_label(item, "example label") for item in self.labels)
        if not 1 <= len(labels) <= MAX_OPTIONS:
            raise DecisionError(f"examples require 1 to {MAX_OPTIONS} labels")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "labels", labels)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> GLiClassExample:
        """Build an example from the ``{"text": ..., "labels": [...]}`` shape.

        Args:
            payload: Mapping with ``text`` and ``labels`` keys, the shape
                ``gliclass`` accepts for ``examples``.

        Returns:
            The validated example.

        Raises:
            DecisionError: If the mapping has missing or unsupported keys.
        """

        if set(payload) != {"text", "labels"}:
            raise DecisionError("examples require exactly text and labels")
        labels = payload["labels"]
        if isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence):
            raise TypeError("example labels must be a sequence of strings")
        return cls(text=payload["text"], labels=tuple(labels))


@dataclass(frozen=True, slots=True)
class GLiClassLogits:
    """Per-label logits from one GLiClass forward pass.

    Attributes:
        logits: Logits that drive the decision, one per label in label order.
            For calibrated checkpoints these are the calibrated logits.
        uncalibrated_logits: Raw logits before the calibrator, when the runtime
            exposes them; ``None`` otherwise.
        token_count: Number of input tokens after formatting, when known.
    """

    logits: tuple[float, ...]
    uncalibrated_logits: tuple[float, ...] | None = None
    token_count: int | None = None

    def __post_init__(self) -> None:
        logits = _finite_values(self.logits, "logits")
        if not logits:
            raise DecisionConflictError("GLiClass returned no logits")
        uncalibrated = (
            None
            if self.uncalibrated_logits is None
            else _finite_values(self.uncalibrated_logits, "uncalibrated logits")
        )
        if uncalibrated is not None and len(uncalibrated) != len(logits):
            raise DecisionConflictError(
                "GLiClass calibrated and uncalibrated logits differ in length"
            )
        if self.token_count is not None and (
            type(self.token_count) is not int or self.token_count < 0
        ):
            raise DecisionConflictError("token_count must be a non-negative integer")
        object.__setattr__(self, "logits", logits)
        object.__setattr__(self, "uncalibrated_logits", uncalibrated)


@runtime_checkable
class GLiClassRunner(Protocol):
    """Runtime adapter that returns GLiClass logits for one text."""

    runtime: str

    def classify(
        self,
        text: str,
        labels: Sequence[str],
        *,
        prompt: str | None = None,
        examples: Sequence[GLiClassExample] = (),
    ) -> GLiClassLogits:
        """Return one logit per label for ``text``."""


def format_gliclass_input(
    text: str,
    labels: Sequence[str],
    *,
    prompt: str | None = None,
    examples: Sequence[GLiClassExample] = (),
    prompt_first: bool = True,
) -> str:
    """Render the uni-encoder input string used by ``gliclass`` 0.1.21.

    The layout is ``<<LABEL>>label ... <<SEP>>`` followed by the instruction
    prompt, then the text and the few-shot examples when ``prompt_first`` is
    true (GLiClass v4). Older checkpoints place the text first.

    Args:
        text: Input text to classify.
        labels: Labels in scoring order.
        prompt: Optional instruction prompt.
        examples: Optional few-shot examples.
        prompt_first: Whether the checkpoint was trained with the label and
            prompt block before the text.

    Returns:
        The formatted string. It contains the input text, so callers must not
        log it.
    """

    head = "".join(f"{_LABEL_TOKEN}{label}" for label in labels) + _SEP_TOKEN
    if prompt:
        head += prompt
    rendered_examples = ""
    if examples:
        rendered_examples = "".join(_render_example(item) for item in examples)
        rendered_examples += _SEP_TOKEN
    if prompt_first:
        return head + text + rendered_examples
    return text + head + rendered_examples


def _render_example(example: GLiClassExample) -> str:
    labels = ", ".join(example.labels)
    return f"{_EXAMPLE_TOKEN}{example.text} \nLabels:\n {labels}"


class GLiClassTorchRunner:
    """Run a ``gliclass`` uni-encoder pipeline on torch.

    The runner reuses the pipeline's own input preparation (labels, prompt
    placement and example tokens) and reads both the calibrated ``logits`` and
    the ``uncalibrated_logits`` of the model output in one forward pass, the
    same values ``ZeroShotClassificationPipeline`` reports as ``score`` and
    ``initial_score``.

    Args:
        pipeline: A ``gliclass.ZeroShotClassificationPipeline`` (or its inner
            uni-encoder pipeline) with ``prepare_inputs`` and ``model``.
        max_length: Token budget. Inputs that reach it are refused instead of
            being truncated.
    """

    runtime = "torch"
    variant: str | None = None

    def __init__(self, pipeline: Any, *, max_length: int | None = None) -> None:
        pipe = getattr(pipeline, "pipe", pipeline)
        if not callable(getattr(pipe, "prepare_inputs", None)) or not callable(
            getattr(pipe, "model", None)
        ):
            raise TypeError(
                "pipeline must be a gliclass ZeroShotClassificationPipeline"
            )
        self._pipe = pipe
        pipe_limit = getattr(pipe, "max_length", None)
        if max_length is None:
            max_length = (
                GLICLASS_DEFAULT_MAX_LENGTH if pipe_limit is None else pipe_limit
            )
        limit = _positive_int(max_length, "max_length")
        if type(pipe_limit) is int and 0 < pipe_limit < limit:
            # The pipeline truncates at its own limit; refuse before that point.
            limit = pipe_limit
        self.max_length = limit

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = GLICLASS_V4_EDGE_MODEL_ID,
        *,
        revision: str | None = None,
        local_dir: str | os.PathLike[str] | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        local_files_only: bool = False,
        device: str = "cpu",
        max_length: int = GLICLASS_DEFAULT_MAX_LENGTH,
    ) -> GLiClassTorchRunner:
        """Load a GLiClass uni-encoder checkpoint for torch inference.

        Args:
            model_id: Hugging Face repository id.
            revision: Immutable revision. Defaults to the pinned
                gliclass-v4-edge commit; other checkpoints must pass one.
            local_dir: Pre-downloaded checkpoint directory. When given, no
                network call is made.
            cache_dir: Optional Hugging Face cache directory.
            local_files_only: Refuse network access and use cached files only.
            device: Torch device, ``"cpu"`` by default.
            max_length: Token budget; longer inputs are refused.

        Returns:
            A ready runner.

        Raises:
            MissingDependencyError: If ``openmed[gliclass]`` is not installed.
            DecisionError: If the checkpoint is not a GLiClass uni-encoder.
        """

        torch, model_cls, pipeline_cls = _load_gliclass()
        checkpoint = _checkpoint_dir(
            model_id,
            revision=_resolve_revision(model_id, revision),
            local_dir=local_dir,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            allow_patterns=_TORCH_CHECKPOINT_FILES,
            hint=GLICLASS_TORCH_INSTALL_HINT,
        )
        model = model_cls.from_pretrained(str(checkpoint), dtype=torch.float32)
        architecture = getattr(model.config, "architecture_type", None)
        if architecture != "uni-encoder":
            raise DecisionError(
                "the GLiClass decision backend requires a uni-encoder checkpoint"
            )
        pipeline = pipeline_cls(
            model,
            str(checkpoint),
            classification_type=GLiClassClassificationType.SINGLE_LABEL.value,
            max_length=max_length,
            device=device,
            progress_bar=False,
        )
        return cls(pipeline, max_length=max_length)

    def classify(
        self,
        text: str,
        labels: Sequence[str],
        *,
        prompt: str | None = None,
        examples: Sequence[GLiClassExample] = (),
    ) -> GLiClassLogits:
        """Return calibrated and uncalibrated logits for every label.

        Args:
            text: Input text; never logged.
            labels: Labels in scoring order.
            prompt: Optional instruction prompt.
            examples: Optional few-shot examples.

        Returns:
            Logits for every label.

        Raises:
            DecisionUnsupportedError: If the formatted input reaches the token
                budget.
            DecisionConflictError: If the model output does not cover every
                label or is not finite.
        """

        label_list = list(labels)
        example_payload = [
            {"text": example.text, "labels": list(example.labels)}
            for example in examples
        ] or None
        inputs = self._pipe.prepare_inputs(
            [text],
            label_list,
            same_labels=True,
            examples=example_payload,
            prompt=prompt,
        )
        token_count = _sequence_length(inputs["input_ids"])
        if token_count >= self.max_length:
            raise DecisionUnsupportedError(
                "input reaches the GLiClass token budget and would be truncated"
            )
        with _no_grad():
            output = self._pipe.model(**inputs, max_num_classes=len(label_list))
        uncalibrated = getattr(output, "uncalibrated_logits", None)
        return GLiClassLogits(
            logits=_first_row(getattr(output, "logits", None), len(label_list)),
            uncalibrated_logits=(
                None
                if uncalibrated is None
                else _first_row(uncalibrated, len(label_list))
            ),
            token_count=token_count,
        )


class GLiClassOnnxRunner:
    """Run an exported GLiClass uni-encoder graph through ONNX Runtime.

    The graph takes ``input_ids`` and ``attention_mask`` and returns calibrated
    ``logits`` with one column per ``<<LABEL>>`` token. The gliclass-v4-edge
    ``onnx/model_q8.onnx`` graph does not expose uncalibrated logits, so
    ``initial_score`` values are unavailable on this path.

    Args:
        session: An ``onnxruntime.InferenceSession`` (or compatible object).
        tokenizer: A ``tokenizers.Tokenizer`` loaded from the export's
            ``tokenizer.json``.
        prompt_first: Checkpoint input layout, ``True`` for GLiClass v4.
        max_length: Token budget; longer inputs are refused, never truncated.
        variant: Short artifact name recorded in the backend id.
    """

    runtime = "onnxruntime"

    def __init__(
        self,
        session: Any,
        tokenizer: Any,
        *,
        prompt_first: bool = True,
        max_length: int = GLICLASS_DEFAULT_MAX_LENGTH,
        variant: str | None = None,
    ) -> None:
        input_names = frozenset(item.name for item in session.get_inputs())
        if input_names != _ONNX_INPUTS:
            raise DecisionError(
                "GLiClass ONNX graphs must take exactly input_ids and attention_mask"
            )
        if _ONNX_OUTPUT not in {item.name for item in session.get_outputs()}:
            raise DecisionError("GLiClass ONNX graphs must expose logits")
        if not callable(getattr(tokenizer, "encode", None)):
            raise TypeError("tokenizer must provide encode()")
        for method in ("no_truncation", "no_padding"):
            disable = getattr(tokenizer, method, None)
            if callable(disable):
                disable()
        self._session = session
        self._tokenizer = tokenizer
        self.prompt_first = bool(prompt_first)
        self.max_length = _positive_int(max_length, "max_length")
        self.variant = variant
        token_to_id = getattr(tokenizer, "token_to_id", None)
        self._label_token_id = (
            token_to_id(_LABEL_TOKEN) if callable(token_to_id) else None
        )

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = GLICLASS_V4_EDGE_MODEL_ID,
        *,
        revision: str | None = None,
        local_dir: str | os.PathLike[str] | None = None,
        onnx_file: str = GLICLASS_ONNX_Q8_FILE,
        tokenizer_file: str = GLICLASS_ONNX_TOKENIZER_FILE,
        cache_dir: str | os.PathLike[str] | None = None,
        local_files_only: bool = False,
        max_length: int = GLICLASS_DEFAULT_MAX_LENGTH,
        providers: Sequence[str] = ("CPUExecutionProvider",),
        session_options: Any | None = None,
    ) -> GLiClassOnnxRunner:
        """Load an exported GLiClass graph and tokenizer.

        Only the selected graph, its tokenizer and ``config.json`` are
        downloaded; the default is the int8 ``onnx/model_q8.onnx`` export.

        Args:
            model_id: Hugging Face repository id.
            revision: Immutable revision. Defaults to the pinned
                gliclass-v4-edge commit; other checkpoints must pass one.
            local_dir: Pre-downloaded repository directory. When given, no
                network call is made.
            onnx_file: Graph path inside the repository.
            tokenizer_file: ``tokenizer.json`` path inside the repository.
            cache_dir: Optional Hugging Face cache directory.
            local_files_only: Refuse network access and use cached files only.
            max_length: Token budget; longer inputs are refused.
            providers: ONNX Runtime execution providers in priority order.
            session_options: Optional ``onnxruntime.SessionOptions``.

        Returns:
            A ready runner.

        Raises:
            MissingDependencyError: If ``openmed[onnx-runtime]`` is not
                installed.
            DecisionError: If the files are not a GLiClass uni-encoder export.
        """

        ort, tokenizer_cls = _load_onnx_runtime()
        directory = _checkpoint_dir(
            model_id,
            revision=_resolve_revision(model_id, revision),
            local_dir=local_dir,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            allow_patterns=(onnx_file, tokenizer_file, "config.json"),
            hint=GLICLASS_ONNX_INSTALL_HINT,
        )
        model_path = _contained_file(directory, onnx_file, "ONNX graph")
        tokenizer_path = _contained_file(directory, tokenizer_file, "tokenizer")
        prompt_first = _uni_encoder_prompt_first(directory)
        session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=list(providers),
        )
        tokenizer = tokenizer_cls.from_file(str(tokenizer_path))
        return cls(
            session,
            tokenizer,
            prompt_first=prompt_first,
            max_length=max_length,
            variant=Path(onnx_file).stem,
        )

    def classify(
        self,
        text: str,
        labels: Sequence[str],
        *,
        prompt: str | None = None,
        examples: Sequence[GLiClassExample] = (),
    ) -> GLiClassLogits:
        """Return calibrated logits for every label.

        Args:
            text: Input text; never logged.
            labels: Labels in scoring order.
            prompt: Optional instruction prompt.
            examples: Optional few-shot examples.

        Returns:
            Logits for every label; ``uncalibrated_logits`` is ``None``.

        Raises:
            DecisionUnsupportedError: If the input exceeds the token budget.
            DecisionConflictError: If the encoded label slots or graph output
                do not match the declared labels.
        """

        import numpy as np

        label_list = list(labels)
        encoding = self._tokenizer.encode(
            format_gliclass_input(
                text,
                label_list,
                prompt=prompt,
                examples=examples,
                prompt_first=self.prompt_first,
            )
        )
        input_ids = list(encoding.ids)
        attention_mask = list(encoding.attention_mask)
        if len(input_ids) > self.max_length:
            raise DecisionUnsupportedError(
                "input exceeds the GLiClass token budget and would be truncated"
            )
        if self._label_token_id is not None and input_ids.count(
            self._label_token_id
        ) != len(label_list):
            raise DecisionConflictError(
                "GLiClass label slots do not match the declared labels"
            )
        outputs = self._session.run(
            [_ONNX_OUTPUT],
            {
                "input_ids": np.asarray([input_ids], dtype=np.int64),
                "attention_mask": np.asarray([attention_mask], dtype=np.int64),
            },
        )
        return GLiClassLogits(
            logits=_first_row(outputs[0], len(label_list)),
            token_count=len(input_ids),
        )


@dataclass(frozen=True, slots=True)
class GLiClassDecisionScores:
    """Calibrated and initial GLiClass scores for one decision request.

    Attributes:
        mode: Decision mode that was scored.
        classification_type: GLiClass activation used for the labels.
        runtime: Runtime that produced the logits.
        scores: Calibrated scores per label: the request options, or the score
            levels for ``scalar_score``.
        initial_scores: Uncalibrated scores per label, or ``None`` when the
            runtime does not expose them.
        scalar_score: Expected score level in ``[0, 1]`` for ``scalar_score``.
        initial_scalar_score: The same expectation over ``initial_scores``.
        confidence: Probability of the most likely level for ``scalar_score``.
        token_count: Formatted input length in tokens, when known.
        prompt_digest: SHA-256 digest of the instruction prompt, if any.
        example_count: Number of few-shot examples sent with the request.
    """

    mode: DecisionMode
    classification_type: GLiClassClassificationType
    runtime: str
    scores: tuple[float, ...]
    initial_scores: tuple[float, ...] | None
    scalar_score: float | None
    initial_scalar_score: float | None
    confidence: float | None
    token_count: int | None
    prompt_digest: str | None
    example_count: int

    def backend_output(self, calibration_id: str) -> DecisionBackendOutput:
        """Return the calibrated scores in the decision backend contract.

        Args:
            calibration_id: Calibration profile requested by the caller.

        Returns:
            Option scores, or a scalar score with its confidence.
        """

        if self.mode is DecisionMode.SCALAR_SCORE:
            return DecisionBackendOutput(
                scalar_score=self.scalar_score,
                confidence=self.confidence,
                calibration_id=calibration_id,
            )
        return DecisionBackendOutput(
            option_scores=self.scores,
            calibration_id=calibration_id,
        )

    def metadata(self) -> dict[str, Any]:
        """Return JSON-safe score metadata without text, options or prompts."""

        return {
            "schema": GLICLASS_METADATA_SCHEMA,
            "runtime": self.runtime,
            "classification_type": self.classification_type.value,
            "label_source": (
                "score_levels" if self.mode is DecisionMode.SCALAR_SCORE else "options"
            ),
            "label_count": len(self.scores),
            "scores": list(self.scores),
            "initial_scores": (
                None if self.initial_scores is None else list(self.initial_scores)
            ),
            "initial_scores_available": self.initial_scores is not None,
            "scalar_score": self.scalar_score,
            "initial_scalar_score": self.initial_scalar_score,
            "token_count": self.token_count,
            "prompt_digest": self.prompt_digest,
            "example_count": self.example_count,
        }


class GLiClassDecisionBackend(EncoderDecisionBackend):
    """Encoder decision backend over a GLiClass v4 uni-encoder.

    The instruction prompt, few-shot examples and score levels are backend
    settings because the 1.0 decision request has no per-question
    instructions. Use :meth:`with_instructions` to score another question with
    the same loaded model. The prompt, the examples and the request text are
    never logged or included in results; results carry a prompt digest.

    Args:
        runner: A :class:`GLiClassTorchRunner`, :class:`GLiClassOnnxRunner`
            or another :class:`GLiClassRunner`.
        model_id: Model identifier recorded in the backend identity.
        revision: Immutable model revision recorded in the identity.
        license_id: Permissive SPDX license of the weights.
        backend_id: Backend identifier; derived from the runtime by default.
        prompt: Optional instruction prompt (policy, rubric or question).
        examples: Optional few-shot examples, as :class:`GLiClassExample` or
            ``{"text": ..., "labels": [...]}`` mappings.
        score_levels: Ordered levels, lowest first, used as labels for
            ``scalar_score`` requests.
    """

    def __init__(
        self,
        runner: GLiClassRunner,
        *,
        model_id: str = GLICLASS_V4_EDGE_MODEL_ID,
        revision: str = GLICLASS_V4_EDGE_REVISION,
        license_id: str = GLICLASS_V4_EDGE_LICENSE,
        backend_id: str | None = None,
        prompt: str | None = None,
        examples: Sequence[GLiClassExample | Mapping[str, Any]] = (),
        score_levels: Sequence[str] = GLICLASS_DEFAULT_SCORE_LEVELS,
    ) -> None:
        if not callable(getattr(runner, "classify", None)):
            raise TypeError("runner must provide classify()")
        runtime = getattr(runner, "runtime", None)
        if not isinstance(runtime, str) or not runtime.strip():
            raise TypeError("runner must declare its runtime")
        variant = getattr(runner, "variant", None)
        default_id = f"openmed.gliclass.v4.{runtime}"
        if isinstance(variant, str) and variant:
            default_id = f"{default_id}.{variant}"
        super().__init__(
            backend_id=backend_id or default_id,
            model_id=model_id,
            revision=revision,
            license_id=license_id,
            scorer=self._score_payload,
            runtime=runtime,
        )
        self._runner = runner
        self._prompt = (
            None
            if prompt is None
            else _checked_text(
                prompt,
                "instruction prompt",
                maximum_chars=MAX_PROMPT_CHARS,
                maximum_bytes=MAX_PROMPT_BYTES,
                multiline=True,
            )
        )
        self._examples = _checked_examples(examples)
        self._score_levels = _checked_levels(score_levels)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = GLICLASS_V4_EDGE_MODEL_ID,
        *,
        runtime: str = "torch",
        revision: str | None = None,
        local_dir: str | os.PathLike[str] | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        local_files_only: bool = False,
        max_length: int = GLICLASS_DEFAULT_MAX_LENGTH,
        device: str = "cpu",
        onnx_file: str = GLICLASS_ONNX_Q8_FILE,
        tokenizer_file: str = GLICLASS_ONNX_TOKENIZER_FILE,
        license_id: str = GLICLASS_V4_EDGE_LICENSE,
        prompt: str | None = None,
        examples: Sequence[GLiClassExample | Mapping[str, Any]] = (),
        score_levels: Sequence[str] = GLICLASS_DEFAULT_SCORE_LEVELS,
    ) -> GLiClassDecisionBackend:
        """Load gliclass-v4-edge (or another uni-encoder) as a decision backend.

        Args:
            model_id: Hugging Face repository id recorded in the identity.
            runtime: ``"torch"`` (``openmed[gliclass]``) or ``"onnx"``
                (``openmed[onnx-runtime]``).
            revision: Immutable revision. Defaults to the pinned
                gliclass-v4-edge commit; other checkpoints must pass one.
            local_dir: Pre-downloaded copy of the repository. When given, no
                network call is made and the path is never recorded.
            cache_dir: Optional Hugging Face cache directory.
            local_files_only: Refuse network access and use cached files only.
            max_length: Token budget; longer inputs are refused.
            device: Torch device for the torch runtime.
            onnx_file: Graph path for the ONNX runtime.
            tokenizer_file: Tokenizer path for the ONNX runtime.
            license_id: Permissive SPDX license of the weights.
            prompt: Optional instruction prompt.
            examples: Optional few-shot examples.
            score_levels: Ordered levels for ``scalar_score`` requests.

        Returns:
            A ready decision backend.

        Raises:
            MissingDependencyError: If the runtime's extra is not installed.
            DecisionError: If the runtime or checkpoint is unsupported.
        """

        if runtime not in {"torch", "onnx", "onnxruntime"}:
            raise DecisionError("GLiClass runtime must be 'torch' or 'onnx'")
        pinned = _resolve_revision(model_id, revision)
        if runtime == "torch":
            runner: GLiClassRunner = GLiClassTorchRunner.from_pretrained(
                model_id,
                revision=pinned,
                local_dir=local_dir,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                device=device,
                max_length=max_length,
            )
        else:
            runner = GLiClassOnnxRunner.from_pretrained(
                model_id,
                revision=pinned,
                local_dir=local_dir,
                onnx_file=onnx_file,
                tokenizer_file=tokenizer_file,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                max_length=max_length,
            )
        return cls(
            runner,
            model_id=model_id,
            revision=pinned,
            license_id=license_id,
            prompt=prompt,
            examples=examples,
            score_levels=score_levels,
        )

    @property
    def runner(self) -> GLiClassRunner:
        """Return the runtime adapter shared by derived backends."""

        return self._runner

    @property
    def prompt_digest(self) -> str | None:
        """Return the SHA-256 digest of the instruction prompt, if any."""

        if self._prompt is None:
            return None
        return "sha256:" + hashlib.sha256(self._prompt.encode("utf-8")).hexdigest()

    @property
    def example_count(self) -> int:
        """Return the number of configured few-shot examples."""

        return len(self._examples)

    @property
    def score_levels(self) -> tuple[str, ...]:
        """Return the ordered levels used for ``scalar_score`` requests."""

        return self._score_levels

    def with_instructions(
        self,
        prompt: str | None,
        *,
        examples: Sequence[GLiClassExample | Mapping[str, Any]] | None = None,
        score_levels: Sequence[str] | None = None,
    ) -> GLiClassDecisionBackend:
        """Return a backend for another question that shares this model.

        Args:
            prompt: Instruction prompt for the new question, or ``None``.
            examples: Few-shot examples; keeps the current ones when ``None``.
            score_levels: Scalar score levels; keeps the current ones when
                ``None``.

        Returns:
            A new backend with the same runner and identity.
        """

        return type(self)(
            self._runner,
            model_id=self.identity.model_id,
            revision=self.identity.revision,
            license_id=self.identity.license_id,
            backend_id=self.identity.backend_id,
            prompt=prompt,
            examples=self._examples if examples is None else examples,
            score_levels=self._score_levels if score_levels is None else score_levels,
        )

    def score_details(self, request: DecisionRequest) -> GLiClassDecisionScores:
        """Score one request and keep calibrated and initial scores.

        Args:
            request: A validated decision request.

        Returns:
            Calibrated scores, initial scores when available, and text-free
            scoring metadata.

        Raises:
            DecisionUnsupportedError: If the request carries GLiClass control
                tokens or exceeds the token budget.
            DecisionConflictError: If the runtime output does not match the
                declared labels.
        """

        if not isinstance(request, DecisionRequest):
            raise TypeError("request must be a DecisionRequest")
        return self._details(request.input_text, request.options, request.mode)

    def _score_payload(
        self,
        input_text: str,
        options: tuple[str, ...],
        mode: DecisionMode,
    ) -> dict[str, Any]:
        details = self._details(input_text, options, mode)
        if details.mode is DecisionMode.SCALAR_SCORE:
            return {
                "scalar_score": details.scalar_score,
                "confidence": details.confidence,
            }
        return {"option_scores": details.scores}

    def _details(
        self,
        input_text: str,
        options: Sequence[str],
        mode: DecisionMode,
    ) -> GLiClassDecisionScores:
        mode = DecisionMode(mode)
        if mode is DecisionMode.SCALAR_SCORE:
            labels = self._score_levels
            classification_type = GLiClassClassificationType.SINGLE_LABEL
        else:
            labels = tuple(options)
            classification_type = (
                GLiClassClassificationType.MULTI_LABEL
                if mode is DecisionMode.MULTI_LABEL
                else GLiClassClassificationType.SINGLE_LABEL
            )
        if _has_control_marker(input_text) or any(
            _has_control_marker(label) for label in labels
        ):
            raise DecisionUnsupportedError(
                "GLiClass control tokens are not accepted in decision requests"
            )
        raw = self._runner.classify(
            input_text,
            labels,
            prompt=self._prompt,
            examples=self._examples,
        )
        if not isinstance(raw, GLiClassLogits) or len(raw.logits) != len(labels):
            raise DecisionConflictError(
                "GLiClass runtime output does not match the declared labels"
            )
        activate = (
            _softmax
            if classification_type is GLiClassClassificationType.SINGLE_LABEL
            else _sigmoid_all
        )
        scores = activate(raw.logits)
        initial = (
            None
            if raw.uncalibrated_logits is None
            else activate(raw.uncalibrated_logits)
        )
        scalar = initial_scalar = confidence = None
        if mode is DecisionMode.SCALAR_SCORE:
            scalar = _expected_level(scores)
            initial_scalar = None if initial is None else _expected_level(initial)
            confidence = max(scores)
        return GLiClassDecisionScores(
            mode=mode,
            classification_type=classification_type,
            runtime=self.identity.runtime,
            scores=scores,
            initial_scores=initial,
            scalar_score=scalar,
            initial_scalar_score=initial_scalar,
            confidence=confidence,
            token_count=raw.token_count,
            prompt_digest=self.prompt_digest,
            example_count=len(self._examples),
        )


class _CapturingBackend:
    """One-shot proxy that keeps the GLiClass details of a single decision."""

    def __init__(self, backend: GLiClassDecisionBackend) -> None:
        self.identity: DecisionBackendIdentity = backend.identity
        self._backend = backend
        self.details: GLiClassDecisionScores | None = None

    def score(self, request: DecisionRequest) -> DecisionBackendOutput:
        details = self._backend.score_details(request)
        self.details = details
        return details.backend_output(request.calibration_id)


def decide_with_gliclass(
    request: DecisionRequest | Mapping[str, Any],
    *,
    backend: GLiClassDecisionBackend,
    policy: DecisionAccessPolicy | None = None,
    calibration_profiles: Mapping[str, DecisionCalibrationProfile] | None = None,
) -> DecisionResult:
    """Decide with GLiClass and keep its scores in ``extensions["gliclass"]``.

    Abstention, access policy, calibration lookup and failure mapping are the
    ones in :func:`openmed.structured.decision.decide`. For ``success`` and
    ``abstained`` results the calibrated scores, initial scores (when the
    runtime exposes them), classification type, token count and prompt digest
    are added under ``extensions["gliclass"]``.

    Args:
        request: A decision request or its mapping form.
        backend: The GLiClass backend to score with.
        policy: Optional access policy.
        calibration_profiles: Optional calibration profiles by ID.

    Returns:
        The decision result.
    """

    if not isinstance(backend, GLiClassDecisionBackend):
        raise TypeError("backend must be a GLiClassDecisionBackend")
    normalized = (
        request
        if isinstance(request, DecisionRequest)
        else DecisionRequest.from_dict(request)
    )
    capture = _CapturingBackend(backend)
    result = decide(
        normalized,
        backend=capture,
        policy=policy,
        calibration_profiles=calibration_profiles,
    )
    if capture.details is None or result.state not in {
        DecisionState.SUCCESS,
        DecisionState.ABSTAINED,
    }:
        return result
    extensions = dict(result.extensions)
    extensions["gliclass"] = capture.details.metadata()
    return replace(result, extensions=extensions)


def _load_gliclass() -> tuple[Any, Any, Any]:
    try:
        import torch
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline
    except ImportError as exc:
        raise MissingDependencyError(
            dependency=_missing_module(exc, "gliclass"),
            instruction=GLICLASS_TORCH_INSTALL_HINT,
        ) from exc
    return torch, GLiClassModel, ZeroShotClassificationPipeline


def _load_onnx_runtime() -> tuple[Any, Any]:
    try:
        import numpy  # noqa: F401
        import onnxruntime
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise MissingDependencyError(
            dependency=_missing_module(exc, "onnxruntime"),
            instruction=GLICLASS_ONNX_INSTALL_HINT,
        ) from exc
    return onnxruntime, Tokenizer


def _missing_module(exc: ImportError, default: str) -> str:
    name = getattr(exc, "name", None)
    return name.split(".", 1)[0] if isinstance(name, str) and name else default


def _resolve_revision(model_id: str, revision: str | None) -> str:
    if revision is None:
        if model_id != GLICLASS_V4_EDGE_MODEL_ID:
            raise DecisionError(
                "pass an immutable revision for GLiClass checkpoints other than "
                "gliclass-v4-edge"
            )
        return GLICLASS_V4_EDGE_REVISION
    if not isinstance(revision, str) or not revision.strip() or len(revision) > 256:
        raise DecisionError("revision must be bounded non-empty text")
    return revision


def _checkpoint_dir(
    model_id: str,
    *,
    revision: str,
    local_dir: str | os.PathLike[str] | None,
    cache_dir: str | os.PathLike[str] | None,
    local_files_only: bool,
    allow_patterns: Sequence[str],
    hint: str,
) -> Path:
    if local_dir is not None:
        directory = Path(local_dir).expanduser()
        if not directory.is_dir():
            raise FileNotFoundError("GLiClass local_dir is not a directory")
        return directory
    if not isinstance(model_id, str) or not model_id.strip():
        raise DecisionError("model_id must be a Hugging Face repository id")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise MissingDependencyError(
            dependency="huggingface_hub",
            instruction=hint,
        ) from exc
    return Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=None if cache_dir is None else str(cache_dir),
            local_files_only=local_files_only,
            allow_patterns=list(allow_patterns),
        )
    )


def _contained_file(directory: Path, relative: str, name: str) -> Path:
    # Hub snapshots are symlinks into a blob store, so containment is checked
    # on the relative path instead of the resolved target.
    path = Path(relative)
    if not relative or path.anchor or ".." in path.parts:
        raise DecisionError(f"GLiClass {name} path must stay inside the checkpoint")
    candidate = directory / path
    if not candidate.is_file():
        raise FileNotFoundError(f"GLiClass {name} file is missing")
    return candidate


def _uni_encoder_prompt_first(directory: Path) -> bool:
    config_path = directory / "config.json"
    if not config_path.is_file():
        return True
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, Mapping):
        raise DecisionError("GLiClass config.json must be an object")
    if config.get("architecture_type", "uni-encoder") != "uni-encoder":
        raise DecisionError(
            "the GLiClass decision backend requires a uni-encoder checkpoint"
        )
    return bool(config.get("prompt_first", True))


def _no_grad() -> contextlib.AbstractContextManager[Any]:
    torch = sys.modules.get("torch")
    no_grad = getattr(torch, "no_grad", None)
    return no_grad() if callable(no_grad) else contextlib.nullcontext()


def _sequence_length(input_ids: Any) -> int:
    shape = getattr(input_ids, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[-1])
    try:
        return len(input_ids[0])
    except (IndexError, TypeError):
        raise DecisionConflictError("GLiClass inputs have an invalid shape") from None


def _first_row(values: Any, count: int) -> tuple[float, ...]:
    if values is None:
        raise DecisionConflictError("GLiClass runtime returned no logits")
    try:
        row = values[0]
        detach = getattr(row, "detach", None)
        if callable(detach):
            row = detach().float().cpu()
        to_list = getattr(row, "tolist", None)
        items = to_list() if callable(to_list) else list(row)
        if len(items) < count:
            raise DecisionConflictError("GLiClass logits do not cover every label")
        return _finite_values(items[:count], "logits")
    except (IndexError, TypeError, ValueError) as exc:
        if isinstance(exc, DecisionError):
            raise
        raise DecisionConflictError("GLiClass logits have an invalid shape") from None


def _finite_values(values: Any, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise DecisionConflictError(f"GLiClass {name} must be a sequence")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise DecisionConflictError(f"GLiClass {name} must be numbers")
        number = float(value)
        if not math.isfinite(number):
            raise DecisionConflictError(f"GLiClass {name} must be finite")
        result.append(number)
    return tuple(result)


def _softmax(logits: Sequence[float]) -> tuple[float, ...]:
    peak = max(logits)
    exps = [math.exp(value - peak) for value in logits]
    total = math.fsum(exps)
    return tuple(_unit(value / total) for value in exps)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return _unit(1.0 / (1.0 + math.exp(-value)))
    exp = math.exp(value)
    return _unit(exp / (1.0 + exp))


def _sigmoid_all(logits: Sequence[float]) -> tuple[float, ...]:
    return tuple(_sigmoid(value) for value in logits)


def _unit(value: float) -> float:
    return min(1.0, max(0.0, value))


def _expected_level(probabilities: Sequence[float]) -> float:
    span = len(probabilities) - 1
    expected = math.fsum(index * value for index, value in enumerate(probabilities))
    return _unit(expected / span)


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise DecisionError(f"{name} must be a positive integer")
    return value


def _has_control_marker(value: str) -> bool:
    return any(marker in value for marker in _CONTROL_MARKERS)


def _checked_text(
    value: Any,
    name: str,
    *,
    maximum_chars: int,
    maximum_bytes: int,
    multiline: bool,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise DecisionError(f"{name} must not be blank")
    if len(value) > maximum_chars or len(value.encode("utf-8")) > maximum_bytes:
        raise DecisionError(f"{name} exceeds its size limit")
    allowed = {"\n", "\r", "\t"} if multiline else set()
    if any(
        (ord(char) < 32 and char not in allowed)
        or ord(char) == 127
        or 0xD800 <= ord(char) <= 0xDFFF
        for char in value
    ):
        raise DecisionError(f"{name} contains invalid characters")
    if _has_control_marker(value):
        raise DecisionError(f"{name} cannot contain GLiClass control tokens")
    return value


def _checked_label(value: Any, name: str) -> str:
    return _checked_text(
        value.strip() if isinstance(value, str) else value,
        name,
        maximum_chars=MAX_OPTION_CHARS,
        maximum_bytes=MAX_OPTION_BYTES,
        multiline=False,
    )


def _checked_examples(
    values: Sequence[GLiClassExample | Mapping[str, Any]],
) -> tuple[GLiClassExample, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("examples must be a sequence")
    examples = tuple(
        item
        if isinstance(item, GLiClassExample)
        else GLiClassExample.from_mapping(item)
        for item in values
    )
    if len(examples) > MAX_EXAMPLES:
        raise DecisionError(f"GLiClass accepts at most {MAX_EXAMPLES} examples")
    return examples


def _checked_levels(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("score levels must be a sequence of strings")
    levels = tuple(_checked_label(item, "score level") for item in values)
    if not 2 <= len(levels) <= MAX_SCORE_LEVELS:
        raise DecisionError(f"score levels require 2 to {MAX_SCORE_LEVELS} entries")
    if len({item.casefold() for item in levels}) != len(levels):
        raise DecisionError("score levels must be unique")
    return levels


__all__ = [
    "GLICLASS_DEFAULT_MAX_LENGTH",
    "GLICLASS_DEFAULT_SCORE_LEVELS",
    "GLICLASS_ONNX_INSTALL_HINT",
    "GLICLASS_ONNX_Q8_FILE",
    "GLICLASS_ONNX_TOKENIZER_FILE",
    "GLICLASS_TORCH_INSTALL_HINT",
    "GLICLASS_V4_EDGE_LICENSE",
    "GLICLASS_V4_EDGE_MODEL_ID",
    "GLICLASS_V4_EDGE_REVISION",
    "GLiClassClassificationType",
    "GLiClassDecisionBackend",
    "GLiClassDecisionScores",
    "GLiClassExample",
    "GLiClassLogits",
    "GLiClassOnnxRunner",
    "GLiClassRunner",
    "GLiClassTorchRunner",
    "decide_with_gliclass",
    "format_gliclass_input",
]
