"""Optional CoNLL-style Indic NER adapter with exact character offsets."""

from __future__ import annotations

import hashlib
import importlib
import logging
import math
import os
import re
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core.labels import LOCATION, ORGANIZATION, PERSON, normalize_label
from openmed.core.model_registry import (
    INDIC_ENCODER_SPECS,
    IndicEncoderLoadResult,
    IndicEncoderSpec,
    get_indic_encoder_spec,
)
from openmed.core.pii_i18n import INDIC_NER_MODEL_ENV

from ..exceptions import MissingDependencyError
from .base import EncoderOutput

_NER_INSTALL_HINT = "Run `pip install .[hf]` to enable the optional Indic NER adapter."
_ENCODER_INSTALL_HINT = (
    "Install the Hugging Face extra and a PyTorch runtime, for example "
    "`pip install 'openmed[hf]' torch`."
)
_TAG_RE = re.compile(r"^([BIESUL])[-_](.+)$", re.IGNORECASE)
_SUPPORTED_LABELS = frozenset({PERSON, LOCATION, ORGANIZATION})
logger = logging.getLogger(__name__)


class IndicNerWeightsUnavailable(RuntimeError):
    """Raised when no user-supplied Indic NER model is configured."""


class IndicNerCheckpointUnavailable(RuntimeError):
    """Raised when explicitly configured Indic NER weights cannot be loaded."""


class IndicNerCompatibilityError(ValueError):
    """Raised when an Indic NER checkpoint violates the safe adapter contract."""


@dataclass(frozen=True)
class IndicNerPrediction:
    """One PHI-safe prediction containing offsets and labels, never raw text."""

    start: int
    end: int
    label: str
    confidence: float

    @property
    def canonical_label(self) -> str:
        """Return the OpenMed canonical label."""

        return self.label

    def to_dict(self) -> dict[str, int | float | str]:
        """Return an aggregate-safe serialization without entity text."""

        return {
            "confidence": self.confidence,
            "end": self.end,
            "label": self.label,
            "start": self.start,
        }


@dataclass
class IndicNerAdapter:
    """Run a fast-tokenizer CoNLL-2003 token-classification checkpoint."""

    model_id: str
    tokenizer: Any
    model: Any

    def __post_init__(self) -> None:
        config = getattr(self.model, "config", None)
        self._id2label = _compatible_label_mapping(config)
        self._label_count = len(self._id2label)

        num_labels = getattr(config, "num_labels", None)
        if num_labels is not None and (
            not isinstance(num_labels, int)
            or isinstance(num_labels, bool)
            or num_labels != self._label_count
        ):
            raise IndicNerCompatibilityError(
                "Indic NER model config num_labels does not match its label map"
            )

    def predict(
        self,
        text: str,
        *,
        max_length: int | None = None,
    ) -> list[IndicNerPrediction]:
        """Predict PER/LOC/ORG spans while preserving source character offsets."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text:
            return []

        tokenizer_kwargs: dict[str, Any] = {
            "return_offsets_mapping": True,
            "return_tensors": "pt",
            "truncation": True,
        }
        if max_length is not None:
            if not isinstance(max_length, int) or isinstance(max_length, bool):
                raise TypeError("max_length must be an integer")
            if max_length <= 0:
                raise ValueError("max_length must be positive")
            tokenizer_kwargs["max_length"] = max_length
        encoded = _safe_checkpoint_call(
            "Indic NER tokenizer failed while processing input",
            lambda: self.tokenizer(text, **tokenizer_kwargs),
        )
        model_inputs = _safe_checkpoint_call(
            "Indic NER tokenizer returned an incompatible encoding",
            lambda: dict(encoded),
        )
        offsets = model_inputs.pop("offset_mapping", None)
        if offsets is None:
            raise IndicNerCompatibilityError(
                "Indic NER requires a fast tokenizer with offset mappings"
            )

        outputs = _safe_checkpoint_call(
            "Indic NER checkpoint failed during inference",
            lambda: self.model(**model_inputs),
        )
        logits = getattr(outputs, "logits", None)
        if logits is None and isinstance(outputs, Mapping):
            logits = outputs.get("logits")
        if logits is None:
            raise IndicNerCompatibilityError(
                "Indic NER model output must include logits"
            )

        offset_rows = _safe_checkpoint_call(
            "Indic NER tokenizer offsets are incompatible",
            lambda: _first_batch(_to_python(offsets)),
        )
        logit_rows = _safe_checkpoint_call(
            "Indic NER checkpoint logits are incompatible",
            lambda: _first_batch(_to_python(logits)),
        )
        if len(offset_rows) != len(logit_rows):
            raise IndicNerCompatibilityError(
                "Indic NER tokenizer offsets and logits have different lengths"
            )

        tagged_tokens: list[tuple[int, int, str, str, float]] = []
        previous_end = 0
        for raw_offset, raw_logits in zip(offset_rows, logit_rows):
            start, end = _validated_token_offset(raw_offset, text_length=len(text))
            if start == end == 0:
                continue
            if start < previous_end:
                raise IndicNerCompatibilityError(
                    "Indic NER tokenizer offsets overlap or move backwards"
                )
            previous_end = end
            if (
                not isinstance(raw_logits, Sequence)
                or isinstance(raw_logits, (str, bytes))
                or len(raw_logits) != self._label_count
            ):
                raise IndicNerCompatibilityError(
                    "Indic NER checkpoint logit width does not match its label map"
                )
            label_id, confidence = _safe_checkpoint_call(
                "Indic NER checkpoint logits are incompatible",
                lambda: _argmax_with_confidence(raw_logits),
            )
            source_label = self._id2label[label_id]
            tag = _canonical_tag(source_label)
            if tag is None:
                tagged_tokens.append((start, end, "O", "O", confidence))
                continue
            prefix, canonical = tag
            tagged_tokens.append((start, end, prefix, canonical, confidence))

        return _merge_tagged_tokens(tagged_tokens)


def configured_indic_ner_model(model_path: str | None = None) -> str | None:
    """Return an explicitly supplied model path/repo, or the configured env value."""

    value = (
        model_path if model_path is not None else os.environ.get(INDIC_NER_MODEL_ENV)
    )
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def is_indic_ner_configured(model_path: str | None = None) -> bool:
    """Return whether optional Indic NER weights were explicitly configured."""

    return configured_indic_ner_model(model_path) is not None


def load_indic_ner_adapter(
    model_path: str | None = None,
    *,
    cache_dir: str | None = None,
    token: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
) -> IndicNerAdapter:
    """Load a user-configured local path or model repo without a bundled default.

    Existing filesystem paths are always loaded locally. Repository identifiers
    can resolve remotely only after the caller explicitly supplies one.
    """

    model_id = configured_indic_ner_model(model_path)
    if model_id is None:
        raise IndicNerWeightsUnavailable(
            f"{INDIC_NER_MODEL_ENV} is not configured; optional Indic NER weights "
            "were not loaded"
        )
    if not isinstance(local_files_only, bool):
        raise TypeError("local_files_only must be a boolean")
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise MissingDependencyError("transformers", _NER_INSTALL_HINT) from exc

    tokenizer_loader = getattr(transformers, "AutoTokenizer", None)
    model_loader = getattr(transformers, "AutoModelForTokenClassification", None)
    if not callable(getattr(tokenizer_loader, "from_pretrained", None)) or not callable(
        getattr(model_loader, "from_pretrained", None)
    ):
        raise IndicNerCompatibilityError(
            "installed transformers package lacks token-classification loaders"
        )

    load_kwargs: dict[str, Any] = {
        "local_files_only": local_files_only or _is_existing_path(model_id),
        "trust_remote_code": False,
    }
    if cache_dir is not None:
        load_kwargs["cache_dir"] = cache_dir
    if token:
        load_kwargs["token"] = token
    if revision:
        load_kwargs["revision"] = revision

    tokenizer = _load_checkpoint_component(
        tokenizer_loader,
        model_id,
        component="tokenizer",
        use_fast=True,
        **load_kwargs,
    )
    if getattr(tokenizer, "is_fast", False) is not True:
        raise IndicNerCompatibilityError(
            "Indic NER requires a fast tokenizer for exact offsets"
        )
    model = _load_checkpoint_component(
        model_loader,
        model_id,
        component="model",
        **load_kwargs,
    )
    if callable(getattr(model, "eval", None)):
        _safe_checkpoint_call(
            "Indic NER checkpoint failed while entering evaluation mode",
            model.eval,
        )
    return IndicNerAdapter(model_id=model_id, tokenizer=tokenizer, model=model)


def _load_checkpoint_component(
    loader: Any,
    model_id: str,
    *,
    component: str,
    **kwargs: Any,
) -> Any:
    try:
        return loader.from_pretrained(model_id, **kwargs)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        failure_type = type(exc).__name__
    raise IndicNerCheckpointUnavailable(
        f"configured Indic NER {component} could not be loaded ({failure_type})"
    )


def _compatible_label_mapping(config: Any) -> dict[int, str]:
    candidates: list[dict[int, str]] = []
    raw_id2label = getattr(config, "id2label", None)
    raw_label2id = getattr(config, "label2id", None)

    if isinstance(raw_id2label, Mapping) and raw_id2label:
        try:
            candidates.append(_normalize_id2label(raw_id2label))
        except IndicNerCompatibilityError:
            pass
    if isinstance(raw_label2id, Mapping) and raw_label2id:
        try:
            candidates.append(_normalize_label2id(raw_label2id))
        except IndicNerCompatibilityError:
            pass

    compatible = [mapping for mapping in candidates if _is_conll_mapping(mapping)]
    if not compatible:
        raise IndicNerCompatibilityError(
            "Indic NER model config must define a compatible id2label or label2id "
            "map for O, PER, LOC, and ORG"
        )
    if any(mapping != compatible[0] for mapping in compatible[1:]):
        raise IndicNerCompatibilityError(
            "Indic NER model config defines conflicting label maps"
        )
    return compatible[0]


def _normalize_id2label(mapping: Mapping[Any, Any]) -> dict[int, str]:
    normalized: dict[int, str] = {}
    for raw_index, raw_label in mapping.items():
        index = _label_index(raw_index)
        label = _label_name(raw_label)
        if index in normalized:
            raise IndicNerCompatibilityError(
                "Indic NER model config contains duplicate label indices"
            )
        normalized[index] = label
    _validate_contiguous_indices(normalized)
    return normalized


def _normalize_label2id(mapping: Mapping[Any, Any]) -> dict[int, str]:
    normalized: dict[int, str] = {}
    for raw_label, raw_index in mapping.items():
        label = _label_name(raw_label)
        index = _label_index(raw_index)
        if index in normalized:
            raise IndicNerCompatibilityError(
                "Indic NER model config contains duplicate label indices"
            )
        normalized[index] = label
    _validate_contiguous_indices(normalized)
    return normalized


def _label_index(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        index = value
    elif isinstance(value, str) and value.strip().isdigit():
        index = int(value.strip())
    else:
        raise IndicNerCompatibilityError(
            "Indic NER model config label indices must be non-negative integers"
        )
    if index < 0:
        raise IndicNerCompatibilityError(
            "Indic NER model config label indices must be non-negative integers"
        )
    return index


def _label_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IndicNerCompatibilityError(
            "Indic NER model config label names must be non-empty strings"
        )
    return value.strip()


def _validate_contiguous_indices(mapping: Mapping[int, str]) -> None:
    if set(mapping) != set(range(len(mapping))):
        raise IndicNerCompatibilityError(
            "Indic NER model config label indices must be contiguous from zero"
        )


def _is_conll_mapping(mapping: Mapping[int, str]) -> bool:
    has_outside = any(label.upper() == "O" for label in mapping.values())
    canonical_labels = {
        tag[1]
        for label in mapping.values()
        if (tag := _canonical_tag(label)) is not None
    }
    return has_outside and _SUPPORTED_LABELS.issubset(canonical_labels)


def _safe_checkpoint_call(message: str, operation: Any) -> Any:
    try:
        return operation()
    except Exception:
        pass
    raise IndicNerCompatibilityError(message)


def _validated_token_offset(
    value: Any,
    *,
    text_length: int,
) -> tuple[int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise IndicNerCompatibilityError(
            "Indic NER tokenizer returned an invalid character offset"
        )
    start, end = value
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
    ):
        raise IndicNerCompatibilityError(
            "Indic NER tokenizer offsets must use integer boundaries"
        )
    if start == end == 0:
        return 0, 0
    if start < 0 or end <= start or end > text_length:
        raise IndicNerCompatibilityError(
            "Indic NER tokenizer returned an out-of-bounds character offset"
        )
    return start, end


def _canonical_tag(source_label: str) -> tuple[str, str] | None:
    normalized = source_label.strip()
    if not normalized or normalized.upper() == "O":
        return None
    match = _TAG_RE.match(normalized)
    if match is None:
        prefix, entity_label = "S", normalized
    else:
        prefix, entity_label = match.group(1).upper(), match.group(2)
    canonical = normalize_label(entity_label)
    if canonical not in _SUPPORTED_LABELS:
        return None
    return prefix, canonical


def _merge_tagged_tokens(
    tagged_tokens: Sequence[tuple[int, int, str, str, float]],
) -> list[IndicNerPrediction]:
    predictions: list[IndicNerPrediction] = []
    current: list[tuple[int, int, float]] = []
    current_label: str | None = None

    def flush() -> None:
        nonlocal current, current_label
        if current and current_label is not None:
            predictions.append(
                IndicNerPrediction(
                    start=current[0][0],
                    end=current[-1][1],
                    label=current_label,
                    confidence=sum(row[2] for row in current) / len(current),
                )
            )
        current = []
        current_label = None

    for start, end, prefix, canonical, confidence in tagged_tokens:
        if prefix == "O":
            flush()
            continue
        starts_entity = prefix in {"B", "S", "U"}
        if starts_entity or canonical != current_label:
            flush()
        current_label = canonical
        current.append((start, end, confidence))
        if prefix in {"E", "L", "S", "U"}:
            flush()
    flush()
    return predictions


def _to_python(value: Any) -> Any:
    detached = value.detach() if callable(getattr(value, "detach", None)) else value
    cpu_value = detached.cpu() if callable(getattr(detached, "cpu", None)) else detached
    if callable(getattr(cpu_value, "tolist", None)):
        return cpu_value.tolist()
    return cpu_value


def _first_batch(value: Any) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("Indic NER tensors must be sequence-like")
    rows = list(value)
    if len(rows) == 1 and isinstance(rows[0], Sequence):
        return list(rows[0])
    return rows


def _argmax_with_confidence(logits: Any) -> tuple[int, float]:
    if not isinstance(logits, Sequence) or isinstance(logits, (str, bytes)):
        raise ValueError("Indic NER token logits must be sequence-like")
    values = [float(value) for value in logits]
    if not values:
        raise ValueError("Indic NER token logits cannot be empty")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Indic NER token logits must be finite")
    label_id = max(range(len(values)), key=values.__getitem__)
    peak = max(values)
    denominator = sum(math.exp(value - peak) for value in values)
    confidence = math.exp(values[label_id] - peak) / denominator
    return label_id, confidence


@dataclass
class IndicEncoderHandle:
    """Loaded tokenizer/backbone pair implementing the encoder contract."""

    source: str
    metadata: IndicEncoderSpec
    tokenizer: Any
    model: Any
    torch_module: Any
    device: str | None = None

    def encode(self, text: str, *, max_length: int = 512) -> EncoderOutput:
        """Encode one string without logging or returning the raw input text."""

        if not isinstance(text, str):
            raise TypeError("encoder input must be a string")
        if not isinstance(max_length, int) or isinstance(max_length, bool):
            raise TypeError("max_length must be an integer")
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        encoded = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        tokenizer_outputs = dict(encoded)
        raw_offsets = tokenizer_outputs.pop("offset_mapping", None)
        if raw_offsets is None:
            raise ValueError(
                "configured tokenizer does not expose character offset mappings"
            )
        offsets = _normalize_offsets(raw_offsets)
        if any(end > len(text) for _, end in offsets):
            raise ValueError("encoder offsets extend beyond the input text")
        model_inputs = _move_to_device(tokenizer_outputs, self.device)

        inference_mode = getattr(self.torch_module, "inference_mode", None)
        context = inference_mode() if callable(inference_mode) else nullcontext()
        with context:
            model_output = self.model(**model_inputs, return_dict=True)

        hidden_states = _last_hidden_state(model_output)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        output = EncoderOutput(
            tokenizer_outputs=model_inputs,
            offset_mapping=offsets,
            last_hidden_state=hidden_states,
            text_sha256=digest,
        )
        output.validate()
        logger.debug(
            "Encoded Indic input sha256=%s characters=%d tokens=%d",
            digest,
            len(text),
            output.token_count,
        )
        return output


def is_indic_encoder_available() -> bool:
    """Return whether the optional Transformers and PyTorch stack imports."""

    return (
        _optional_module("transformers") is not None
        and _optional_module("torch") is not None
    )


def load_indic_encoder(
    source: str | Path | None,
    *,
    family: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    device: str | None = None,
) -> IndicEncoderLoadResult:
    """Resolve explicit user-supplied MuRIL or IndicBERT weights.

    A missing source, dependency, or checkpoint returns a deterministic skip
    reason. Repository identifiers may download only after the caller supplies
    one explicitly; existing filesystem paths are always loaded locally.
    """

    if source is None:
        return IndicEncoderLoadResult(
            handle=None,
            skip_reason="no Indic encoder weights are configured",
        )
    if not isinstance(source, (str, Path)):
        raise TypeError("encoder source must be a string, path, or None")
    if not isinstance(local_files_only, bool):
        raise TypeError("local_files_only must be a boolean")

    source_value = str(source).strip()
    if not source_value:
        return IndicEncoderLoadResult(
            handle=None,
            skip_reason="no Indic encoder weights are configured",
        )
    metadata = get_indic_encoder_spec(family, source=source_value)
    transformers_module = _optional_module("transformers")
    torch_module = _optional_module("torch")
    if transformers_module is None or torch_module is None:
        missing = "transformers" if transformers_module is None else "torch"
        return IndicEncoderLoadResult(
            handle=None,
            skip_reason=(
                f"optional dependency {missing!r} is unavailable; "
                f"{_ENCODER_INSTALL_HINT}"
            ),
            metadata=metadata,
        )

    tokenizer_loader = getattr(transformers_module, "AutoTokenizer", None)
    model_loader = getattr(transformers_module, "AutoModel", None)
    if tokenizer_loader is None or model_loader is None:
        return IndicEncoderLoadResult(
            handle=None,
            skip_reason="the installed transformers package lacks AutoTokenizer/AutoModel",
            metadata=metadata,
        )

    load_kwargs: dict[str, Any] = {
        "local_files_only": local_files_only or _is_existing_path(source_value),
        "trust_remote_code": False,
    }
    if cache_dir is not None:
        load_kwargs["cache_dir"] = str(cache_dir)
    if token:
        load_kwargs["token"] = token
    if revision:
        load_kwargs["revision"] = revision

    try:
        tokenizer = tokenizer_loader.from_pretrained(
            source_value,
            use_fast=True,
            **load_kwargs,
        )
        if getattr(tokenizer, "is_fast", True) is not True:
            return IndicEncoderLoadResult(
                handle=None,
                skip_reason="configured tokenizer does not provide exact offsets",
                metadata=metadata,
            )
        model = model_loader.from_pretrained(source_value, **load_kwargs)
        if callable(getattr(model, "eval", None)):
            model.eval()
        if device and callable(getattr(model, "to", None)):
            model = model.to(device)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return IndicEncoderLoadResult(
            handle=None,
            skip_reason=(
                f"configured {metadata.display_name} weights could not be loaded "
                f"({type(exc).__name__})"
            ),
            metadata=metadata,
        )

    return IndicEncoderLoadResult(
        handle=IndicEncoderHandle(
            source=source_value,
            metadata=metadata,
            tokenizer=tokenizer,
            model=model,
            torch_module=torch_module,
            device=device,
        ),
        skip_reason=None,
        metadata=metadata,
    )


def _optional_module(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except (ImportError, OSError, RuntimeError, ValueError):
        return None


def _is_existing_path(source: str) -> bool:
    try:
        return Path(source).expanduser().exists()
    except OSError:
        return False


def _normalize_offsets(value: Any) -> tuple[tuple[int, int], ...]:
    if callable(getattr(value, "detach", None)):
        value = value.detach()
    if callable(getattr(value, "cpu", None)):
        value = value.cpu()
    if callable(getattr(value, "tolist", None)):
        value = value.tolist()
    if (
        isinstance(value, (list, tuple))
        and len(value) == 1
        and isinstance(value[0], (list, tuple))
    ):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise ValueError("encoder offsets must be a single batched sequence")
    offsets: list[tuple[int, int]] = []
    for offset in value:
        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            raise ValueError("encoder offsets contain an invalid span")
        start, end = offset
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
        ):
            raise ValueError("encoder offsets must contain integer boundaries")
        offsets.append((start, end))
    return tuple(offsets)


def _move_to_device(
    tokenizer_outputs: Mapping[str, Any],
    device: str | None,
) -> dict[str, Any]:
    if not device:
        return dict(tokenizer_outputs)
    return {
        key: value.to(device) if callable(getattr(value, "to", None)) else value
        for key, value in tokenizer_outputs.items()
    }


def _last_hidden_state(model_output: Any) -> Any:
    hidden_states = getattr(model_output, "last_hidden_state", None)
    if hidden_states is not None:
        return hidden_states
    if isinstance(model_output, Mapping) and "last_hidden_state" in model_output:
        return model_output["last_hidden_state"]
    if isinstance(model_output, (list, tuple)) and model_output:
        return model_output[0]
    raise ValueError("configured encoder did not return last_hidden_state")


__all__ = [
    "INDIC_ENCODER_SPECS",
    "INDIC_NER_MODEL_ENV",
    "IndicEncoderHandle",
    "IndicEncoderLoadResult",
    "IndicEncoderSpec",
    "IndicNerAdapter",
    "IndicNerCheckpointUnavailable",
    "IndicNerCompatibilityError",
    "IndicNerPrediction",
    "IndicNerWeightsUnavailable",
    "configured_indic_ner_model",
    "get_indic_encoder_spec",
    "is_indic_encoder_available",
    "is_indic_ner_configured",
    "load_indic_encoder",
    "load_indic_ner_adapter",
]
