"""Conversion utilities for mapping span entities to token-level labels."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from .infer import Entity


class TokenizerProvider(Protocol):  # pragma: no cover - protocol definition
    """Protocol representing objects that can yield a tokenizer."""

    def get_tokenizer(self) -> Any:
        ...


@dataclass
class TokenAnnotation:
    token: str
    start: int
    end: int
    label: str
    group: Optional[str] = None
    entity: Optional[Entity] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "token": self.token,
            "start": self.start,
            "end": self.end,
            "label": self.label,
        }
        if self.group is not None:
            payload["group"] = self.group
        if self.entity is not None:
            payload["entity"] = self.entity.to_dict()
        return payload


@dataclass
class TokenClassificationResult:
    tokens: List[TokenAnnotation]
    scheme: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def labels(self) -> List[str]:
        return [annotation.label for annotation in self.tokens]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scheme": self.scheme,
            "tokens": [token.to_dict() for token in self.tokens],
            "metadata": self.metadata,
        }


def to_token_classification(
    entities: Sequence[Entity],
    text: str,
    *,
    tokenizer: Any = None,
    scheme: str = "BIO",
) -> TokenClassificationResult:
    """Project entity spans onto token-level labels using ``scheme``."""

    scheme_upper = scheme.upper()
    if scheme_upper not in {"BIO", "BILOU"}:
        raise ValueError(f"Unsupported labelling scheme: {scheme}")

    token_sequences = _tokenize(text, tokenizer)
    assignments = _assign_entities_to_tokens(token_sequences, entities)

    annotations: List[TokenAnnotation] = []
    groups: Dict[str, List[int]] = {}

    for idx, token_info in enumerate(token_sequences):
        token_text, start, end = token_info
        assignment = assignments[idx]
        if assignment is None:
            annotations.append(
                TokenAnnotation(
                    token=token_text,
                    start=start,
                    end=end,
                    label="O",
                )
            )
            continue

        entity, position, span_length = assignment
        label = _label_for_position(entity.label, position, span_length, scheme_upper)
        annotation = TokenAnnotation(
            token=token_text,
            start=start,
            end=end,
            label=label,
            group=entity.group,
            entity=entity,
        )
        annotations.append(annotation)

        if entity.group:
            groups.setdefault(entity.group, []).append(idx)

    metadata = {}
    if groups:
        metadata["groups"] = groups

    return TokenClassificationResult(tokens=annotations, scheme=scheme_upper, metadata=metadata)


def _tokenize(text: str, tokenizer: Any) -> List[Tuple[str, int, int]]:
    if tokenizer is None:
        return _simple_tokenize(text)

    resolved = _resolve_tokenizer(tokenizer)
    if resolved is None:
        return _simple_tokenize(text)

    try:
        encoded = resolved(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
    except TypeError:
        # Some tokenizers require positional arguments only.
        encoded = resolved(text)

    offsets = encoded.get("offset_mapping") if isinstance(encoded, dict) else None
    if not offsets:
        return _simple_tokenize(text)

    tokens: List[Tuple[str, int, int]] = []
    for start, end in offsets:
        if start is None or end is None or start == end:
            continue
        token_text = text[start:end]
        tokens.append((token_text, start, end))
    if tokens:
        return tokens
    return _simple_tokenize(text)


def _resolve_tokenizer(tokenizer: Any):
    if hasattr(tokenizer, "get_tokenizer"):
        return tokenizer.get_tokenizer()
    return tokenizer


def _simple_tokenize(text: str) -> List[Tuple[str, int, int]]:
    tokens: List[Tuple[str, int, int]] = []
    for match in re.finditer(r"\S+", text):
        tokens.append((match.group(0), match.start(), match.end()))
    return tokens


def _assign_entities_to_tokens(
    tokens: Sequence[Tuple[str, int, int]],
    entities: Sequence[Entity],
) -> List[Optional[Tuple[Entity, int, int]]]:
    assignments: List[Optional[Tuple[Entity, int, int]]] = [None] * len(tokens)
    sorted_entities = sorted(entities, key=lambda e: e.score, reverse=True)

    for entity in sorted_entities:
        covered_indices = _token_indices_for_entity(tokens, entity)
        if not covered_indices:
            continue
        span_length = len(covered_indices)
        for position, token_idx in enumerate(covered_indices):
            current = assignments[token_idx]
            if current is None or entity.score > current[0].score:
                assignments[token_idx] = (entity, position, span_length)

    return assignments


def _token_indices_for_entity(
    tokens: Sequence[Tuple[str, int, int]],
    entity: Entity,
) -> List[int]:
    indices: List[int] = []
    for idx, (_, start, end) in enumerate(tokens):
        if _overlaps(start, end, entity.start, entity.end):
            indices.append(idx)
    return indices


def _overlaps(
    token_start: int,
    token_end: int,
    entity_start: int,
    entity_end: int,
) -> bool:
    return token_start < entity_end and token_end > entity_start


def _label_for_position(label: str, position: int, span_length: int, scheme: str) -> str:
    if scheme == "BIO":
        prefix = "B" if position == 0 else "I"
        return f"{prefix}-{label}"

    # BILOU
    if span_length == 1:
        prefix = "U"
    elif position == 0:
        prefix = "B"
    elif position == span_length - 1:
        prefix = "L"
    else:
        prefix = "I"
    return f"{prefix}-{label}"


__all__ = [
    "TokenizerProvider",
    "TokenAnnotation",
    "TokenClassificationResult",
    "to_token_classification",
]
