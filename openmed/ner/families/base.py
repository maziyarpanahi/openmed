"""Shared definitions for model family integrations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol


class ModelFamily(str, Enum):
    """Enumeration of supported model families."""

    GLINER = "gliner"
    GLINER2 = "gliner2"
    INDIC_NER = "indic_ner"
    MURIL = "muril"
    INDICBERT = "indicbert"
    OTHER = "other"


@dataclass(frozen=True)
class EncoderOutput:
    """Stable output contract for backbone-only text encoders.

    The contract intentionally omits the source text. Consumers receive token
    IDs, an attention mask, character offsets, hidden states, and a one-way
    digest that can be used for diagnostics without persisting raw input.
    """

    tokenizer_outputs: Mapping[str, Any]
    offset_mapping: tuple[tuple[int, int], ...]
    last_hidden_state: Any
    text_sha256: str

    @property
    def input_ids(self) -> Any:
        """Return the tokenizer input IDs."""

        return self.tokenizer_outputs["input_ids"]

    @property
    def attention_mask(self) -> Any:
        """Return the tokenizer attention mask."""

        return self.tokenizer_outputs["attention_mask"]

    @property
    def token_count(self) -> int:
        """Return the aligned sequence length, including special tokens."""

        return len(self.offset_mapping)

    @property
    def hidden_size(self) -> int:
        """Return the final hidden-state width."""

        return _shape(self.last_hidden_state)[-1]

    def validate(self) -> None:
        """Raise ``ValueError`` when tensor shapes or offsets are misaligned."""

        missing = {"input_ids", "attention_mask"} - set(self.tokenizer_outputs)
        if missing:
            raise ValueError(
                "encoder tokenizer outputs are missing: " + ", ".join(sorted(missing))
            )

        input_shape = _shape(self.input_ids)
        mask_shape = _shape(self.attention_mask)
        hidden_shape = _shape(self.last_hidden_state)
        if len(input_shape) != 2 or input_shape[0] != 1:
            raise ValueError("encoder input_ids must have shape [1, sequence]")
        if mask_shape != input_shape:
            raise ValueError("encoder attention_mask must align with input_ids")
        if len(hidden_shape) != 3 or hidden_shape[:2] != input_shape:
            raise ValueError(
                "encoder last_hidden_state must have shape [1, sequence, hidden]"
            )
        if len(self.offset_mapping) != input_shape[1]:
            raise ValueError("encoder offsets must align with the token sequence")
        if any(start < 0 or end < start for start, end in self.offset_mapping):
            raise ValueError("encoder offsets must be ordered, non-negative spans")
        if len(self.text_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.text_sha256
        ):
            raise ValueError("encoder text_sha256 must be a lowercase SHA-256 digest")


def _shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(dimension) for dimension in shape)
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        return (len(value), *_shape(value[0]))
    return ()


class SupportsPrediction(Protocol):  # pragma: no cover - interface only
    """Protocol for minimal predictor objects."""

    def predict(self, text: str, *, labels: list[str] | None = None) -> object: ...


class SupportsEncoding(Protocol):  # pragma: no cover - interface only
    """Protocol implemented by reusable tokenizer/backbone handles."""

    tokenizer: Any

    def encode(self, text: str, *, max_length: int = 512) -> EncoderOutput: ...


__all__ = [
    "EncoderOutput",
    "ModelFamily",
    "SupportsEncoding",
    "SupportsPrediction",
]
