"""Local retrieval and explicit privacy-gateway external-model boundary."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .redacted_index import RedactedChunk, RedactedIndex

_SEARCH_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


class ExternalLLMGatewayProxy(Protocol):
    """Explicit gateway boundary for a redacted external-model request."""

    def complete_redacted(self, payload: Mapping[str, Any]) -> Any:
        """Forward a payload that already contains redacted text only."""

        ...


@dataclass(frozen=True)
class RetrievedPassage:
    """A redacted passage safe to hand to a generic external model."""

    text: str
    document_key: str
    chunk_id: str
    score: float
    placeholders: tuple[str, ...]

    def external_payload(self) -> dict[str, Any]:
        """Return only redacted text and PHI-free provenance fields."""

        return {
            "text": self.text,
            "document_key": self.document_key,
            "chunk_id": self.chunk_id,
            "score": self.score,
            "placeholders": list(self.placeholders),
        }


@dataclass(frozen=True)
class RedactedLLMResponse:
    """Redacted response plus the vault keys permitted for restoration."""

    text: str
    document_keys: tuple[str, ...]


class InMemoryVectorStore:
    """Dependency-free sparse-vector store used only in local process memory."""

    def search(
        self,
        query: str,
        chunks: Sequence[RedactedChunk],
        *,
        k: int,
    ) -> tuple[tuple[float, RedactedChunk], ...]:
        """Rank redacted chunks by cosine similarity of binary token vectors."""

        query_tokens = _tokenize(query)
        if not query_tokens:
            return ()
        scored = [
            (_similarity(query_tokens, _tokenize(chunk.text)), chunk)
            for chunk in chunks
        ]
        matched = [item for item in scored if item[0] > 0.0]
        matched.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return tuple(matched[:k])


class RedactedRetriever:
    """Query a :class:`RedactedIndex` through a local vector store."""

    def __init__(
        self,
        index: RedactedIndex,
        *,
        vector_store: InMemoryVectorStore | None = None,
    ) -> None:
        self.index = index
        self.vector_store = vector_store or InMemoryVectorStore()

    def retrieve(
        self,
        query: str,
        *,
        k: int = 4,
    ) -> tuple[RetrievedPassage, ...]:
        """Return the best redacted passages without network egress."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not isinstance(k, int) or isinstance(k, bool):
            raise TypeError("k must be an integer")
        if k < 1:
            raise ValueError("k must be positive")
        scored = self.vector_store.search(query, self.index.chunks, k=k)
        return tuple(
            RetrievedPassage(
                text=chunk.text,
                document_key=chunk.document_key,
                chunk_id=chunk.chunk_id,
                score=score,
                placeholders=chunk.placeholders,
            )
            for score, chunk in scored
        )


class GatewayBoundExternalLLM:
    """Send redacted retrieval payloads only through an explicit gateway proxy."""

    def __init__(self, gateway_proxy: ExternalLLMGatewayProxy) -> None:
        self.gateway_proxy = gateway_proxy

    def invoke(
        self,
        redacted_query: str,
        passages: Sequence[RetrievedPassage],
    ) -> RedactedLLMResponse:
        """Call ``complete_redacted`` on the gateway with a PHI-safe payload."""

        if not isinstance(redacted_query, str):
            raise TypeError("redacted_query must be a string")
        if any(not isinstance(passage, RetrievedPassage) for passage in passages):
            raise TypeError("passages must contain RetrievedPassage values")
        payload = {
            "query": redacted_query,
            "passages": [passage.external_payload() for passage in passages],
        }
        method = getattr(self.gateway_proxy, "complete_redacted", None)
        if not callable(method):
            raise TypeError("gateway_proxy must expose complete_redacted()")
        response = method(payload)
        text = _response_text(response)
        document_keys = tuple(
            dict.fromkeys(passage.document_key for passage in passages)
        )
        return RedactedLLMResponse(text=text, document_keys=document_keys)


def _response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, Mapping):
        for key in ("text", "response", "content", "completion", "output"):
            value = response.get(key)
            if isinstance(value, str):
                return value
    value = getattr(response, "text", None)
    if isinstance(value, str):
        return value
    raise TypeError("privacy-gateway external model response must contain text")


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(
        match.group(0).casefold() for match in _SEARCH_TOKEN_PATTERN.finditer(text)
    )


def _similarity(
    query_tokens: frozenset[str],
    passage_tokens: frozenset[str],
) -> float:
    overlap = len(query_tokens & passage_tokens)
    if overlap == 0 or not passage_tokens:
        return 0.0
    return overlap / math.sqrt(len(query_tokens) * len(passage_tokens))


__all__ = [
    "ExternalLLMGatewayProxy",
    "GatewayBoundExternalLLM",
    "InMemoryVectorStore",
    "RedactedLLMResponse",
    "RedactedRetriever",
    "RetrievedPassage",
]
