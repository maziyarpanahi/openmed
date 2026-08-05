"""Offline index for de-identified clinical note chunks.

The index persists only placeholder-bearing text, HMAC document references,
chunk identifiers, and character offsets. Reversible values are written to a
separate :class:`~openmed.interop.retrieval.EncryptedMappingVault`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openmed.core.labels import CANONICAL_LABELS, OTHER, normalize_label

from .reidentify import PLACEHOLDER_PATTERN, EncryptedMappingVault

INDEX_SCHEMA_VERSION = 1

Deidentifier = Callable[..., Any]

_CHUNK_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_DOCUMENT_KEY_PATTERN = re.compile(r"^hmac-sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class PlaceholderOffset:
    """Offset-only location of one placeholder in a redacted chunk."""

    placeholder: str
    redacted_start: int
    redacted_end: int
    source_start: int
    source_end: int

    def __post_init__(self) -> None:
        if PLACEHOLDER_PATTERN.fullmatch(self.placeholder) is None:
            raise ValueError("placeholder offset contains an invalid token")
        if self.redacted_start < 0 or self.redacted_end <= self.redacted_start:
            raise ValueError("redacted placeholder offsets are invalid")
        if self.source_start < 0 or self.source_end <= self.source_start:
            raise ValueError("source placeholder offsets are invalid")

    def to_dict(self) -> dict[str, int | str]:
        """Return a JSON-compatible offset record with no plaintext PHI."""

        return {
            "placeholder": self.placeholder,
            "redacted_start": self.redacted_start,
            "redacted_end": self.redacted_end,
            "source_start": self.source_start,
            "source_end": self.source_end,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlaceholderOffset":
        """Load and strictly validate one persisted offset record."""

        if set(payload) != {
            "placeholder",
            "redacted_start",
            "redacted_end",
            "source_start",
            "source_end",
        }:
            raise ValueError("placeholder offset contains unsupported fields")
        return cls(
            placeholder=str(payload["placeholder"]),
            redacted_start=int(payload["redacted_start"]),
            redacted_end=int(payload["redacted_end"]),
            source_start=int(payload["source_start"]),
            source_end=int(payload["source_end"]),
        )


@dataclass(frozen=True)
class RedactedChunk:
    """One locally indexed chunk that can never contain a mapping value."""

    chunk_id: str
    document_key: str
    ordinal: int
    text: str
    placeholder_offsets: tuple[PlaceholderOffset, ...] = ()

    def __post_init__(self) -> None:
        if _CHUNK_ID_PATTERN.fullmatch(self.chunk_id) is None:
            raise ValueError("chunk_id must be a SHA-256 reference")
        if _DOCUMENT_KEY_PATTERN.fullmatch(self.document_key) is None:
            raise ValueError("document_key must be an HMAC reference")
        if self.ordinal < 0:
            raise ValueError("chunk ordinal must be non-negative")
        if not isinstance(self.text, str):
            raise TypeError("chunk text must be a string")
        document_digest = self.document_key.rsplit(":", 1)[-1][:16].upper()
        for offset in self.placeholder_offsets:
            if offset.redacted_end > len(self.text):
                raise ValueError("placeholder offset exceeds chunk text")
            if (
                self.text[offset.redacted_start : offset.redacted_end]
                != offset.placeholder
            ):
                raise ValueError("placeholder offset does not match chunk text")
            if f"_{document_digest}_" not in offset.placeholder:
                raise ValueError("placeholder is not bound to its document key")
        discovered = tuple(PLACEHOLDER_PATTERN.findall(self.text))
        recorded = tuple(offset.placeholder for offset in self.placeholder_offsets)
        if discovered != recorded:
            raise ValueError("chunk placeholder offsets are incomplete")

    @property
    def placeholders(self) -> tuple[str, ...]:
        """Return placeholders without opening the mapping vault."""

        return tuple(offset.placeholder for offset in self.placeholder_offsets)

    def to_dict(self) -> dict[str, Any]:
        """Return the complete PHI-free persisted chunk payload."""

        return {
            "chunk_id": self.chunk_id,
            "document_key": self.document_key,
            "ordinal": self.ordinal,
            "text": self.text,
            "placeholder_offsets": [
                offset.to_dict() for offset in self.placeholder_offsets
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RedactedChunk":
        """Load and strictly validate one persisted chunk."""

        if set(payload) != {
            "chunk_id",
            "document_key",
            "ordinal",
            "text",
            "placeholder_offsets",
        }:
            raise ValueError("redacted chunk contains unsupported fields")
        raw_offsets = payload["placeholder_offsets"]
        if not isinstance(raw_offsets, list):
            raise TypeError("placeholder_offsets must be a list")
        return cls(
            chunk_id=str(payload["chunk_id"]),
            document_key=str(payload["document_key"]),
            ordinal=int(payload["ordinal"]),
            text=str(payload["text"]),
            placeholder_offsets=tuple(
                PlaceholderOffset.from_dict(offset) for offset in raw_offsets
            ),
        )


@dataclass(frozen=True)
class IndexingResult:
    """Safe result returned after a document is redacted and indexed."""

    document_key: str
    redacted_text: str
    chunks: tuple[RedactedChunk, ...]

    @property
    def placeholders(self) -> tuple[str, ...]:
        """Return every placeholder in document order."""

        return tuple(
            placeholder for chunk in self.chunks for placeholder in chunk.placeholders
        )


@dataclass(frozen=True)
class _DetectedSpan:
    label: str
    start: int
    end: int


@dataclass(frozen=True)
class _DocumentOffset:
    placeholder: str
    redacted_start: int
    redacted_end: int
    source_start: int
    source_end: int


class RedactedIndex:
    """Thread-safe, zero-egress redacted text index with optional persistence."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._documents: dict[str, tuple[RedactedChunk, ...]] = {}
        self._lock = threading.RLock()
        if self._path is not None and self._path.exists():
            self._load()

    @classmethod
    def from_file(cls, path: str | Path) -> "RedactedIndex":
        """Open or create a file-backed redacted index."""

        return cls(path)

    def index_document(
        self,
        document_id: str,
        text: str,
        *,
        vault: EncryptedMappingVault,
        chunk_size: int = 1000,
        deidentifier: Deidentifier | None = None,
        deidentify_kwargs: Mapping[str, Any] | None = None,
    ) -> IndexingResult:
        """De-identify, vault, chunk, and locally index one clinical note.

        The detector runs locally. Its own rendered text and mapping are ignored;
        exact source offsets are replaced with unique retrieval placeholders so
        each occurrence remains reversible across arbitrary chunk boundaries.
        """

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
            raise TypeError("chunk_size must be an integer")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")

        document_key = vault.document_key(document_id)
        detector = deidentifier or _default_deidentifier
        kwargs = {
            "method": "mask",
            "keep_mapping": False,
            "use_safety_sweep": True,
        }
        kwargs.update(dict(deidentify_kwargs or {}))
        # Retrieval ingestion never asks the core deidentifier to retain or
        # cache a second plaintext mapping.
        kwargs["keep_mapping"] = False
        kwargs["audit"] = False
        kwargs["cache_results"] = False
        result = detector(text, **kwargs)
        spans = _coerce_spans(result, text)
        redacted_text, mapping, offsets = _redact_with_unique_placeholders(
            text,
            spans,
            document_key=document_key,
        )
        stored_key = vault.store(document_id, mapping)
        if stored_key != document_key:
            raise RuntimeError("vault returned an inconsistent document key")
        chunks = _chunk_redacted_document(
            redacted_text,
            offsets,
            document_key=document_key,
            chunk_size=chunk_size,
        )
        with self._lock:
            self._documents[document_key] = chunks
            self._save()
        return IndexingResult(
            document_key=document_key,
            redacted_text=redacted_text,
            chunks=chunks,
        )

    @property
    def chunks(self) -> tuple[RedactedChunk, ...]:
        """Return all redacted chunks in stable document and ordinal order."""

        with self._lock:
            return tuple(
                chunk
                for document_key in sorted(self._documents)
                for chunk in self._documents[document_key]
            )

    @property
    def document_keys(self) -> tuple[str, ...]:
        """Return safe document references present in the index."""

        with self._lock:
            return tuple(sorted(self._documents))

    def chunks_for_document(self, document_key: str) -> tuple[RedactedChunk, ...]:
        """Return redacted chunks for a PHI-safe document reference."""

        with self._lock:
            return tuple(self._documents.get(document_key, ()))

    def _save(self) -> None:
        if self._path is None:
            return
        payload = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "documents": [
                {
                    "document_key": document_key,
                    "chunks": [chunk.to_dict() for chunk in chunks],
                }
                for document_key, chunks in sorted(self._documents.items())
            ],
        }
        _atomic_write_json(self._path, payload)

    def _load(self) -> None:
        assert self._path is not None
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise TypeError("index payload must be an object")
            if set(payload) != {"schema_version", "documents"}:
                raise ValueError("index payload contains unsupported fields")
            if payload["schema_version"] != INDEX_SCHEMA_VERSION:
                raise ValueError("unsupported redacted index schema version")
            documents = payload["documents"]
            if not isinstance(documents, list):
                raise TypeError("index documents must be a list")
            loaded: dict[str, tuple[RedactedChunk, ...]] = {}
            for document in documents:
                if not isinstance(document, Mapping):
                    raise TypeError("index document must be an object")
                if set(document) != {"document_key", "chunks"}:
                    raise ValueError("index document contains unsupported fields")
                document_key = str(document["document_key"])
                if _DOCUMENT_KEY_PATTERN.fullmatch(document_key) is None:
                    raise ValueError("index document key is invalid")
                raw_chunks = document["chunks"]
                if not isinstance(raw_chunks, list):
                    raise TypeError("document chunks must be a list")
                chunks = tuple(RedactedChunk.from_dict(chunk) for chunk in raw_chunks)
                if any(chunk.document_key != document_key for chunk in chunks):
                    raise ValueError("chunk document key does not match its container")
                if tuple(chunk.ordinal for chunk in chunks) != tuple(
                    range(len(chunks))
                ):
                    raise ValueError("chunk ordinals are not contiguous")
                if document_key in loaded:
                    raise ValueError("duplicate index document key")
                loaded[document_key] = chunks
        except (
            KeyError,
            TypeError,
            ValueError,
            UnicodeError,
            json.JSONDecodeError,
        ) as exc:
            raise ValueError("Redacted index failed validation") from exc
        self._documents = loaded

    def __len__(self) -> int:
        return len(self.chunks)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(<{len(self)} redacted chunks>)"


def _default_deidentifier(text: str, **kwargs: Any) -> Any:
    from openmed.core.pii import deidentify

    return deidentify(text, **kwargs)


def _coerce_spans(result: Any, text: str) -> tuple[_DetectedSpan, ...]:
    raw_entities = getattr(result, "pii_entities", None)
    if raw_entities is None:
        raw_entities = getattr(result, "entities", None)
    if raw_entities is None:
        raise TypeError(
            "deidentifier must return an object with pii_entities or entities"
        )
    spans: list[_DetectedSpan] = []
    for entity in raw_entities:
        start = _entity_value(entity, "start")
        end = _entity_value(entity, "end")
        label = _entity_value(entity, "canonical_label", "entity_type", "label")
        try:
            span = _DetectedSpan(
                label=str(label or "UNKNOWN"),
                start=int(start),
                end=int(end),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("detected PHI span metadata is invalid") from exc
        spans.append(span)

    ordered = tuple(sorted(spans, key=lambda span: (span.start, span.end)))
    previous_end = 0
    for span in ordered:
        if span.start < 0 or span.end <= span.start or span.end > len(text):
            raise ValueError("detected PHI span is outside the source text")
        if span.start < previous_end:
            raise ValueError("detected PHI spans overlap")
        previous_end = span.end
    return ordered


def _entity_value(entity: Any, *names: str) -> Any:
    for name in names:
        if isinstance(entity, Mapping) and name in entity:
            value = entity[name]
        else:
            value = getattr(entity, name, None)
        if value is not None:
            return value
    return None


def _redact_with_unique_placeholders(
    text: str,
    spans: Sequence[_DetectedSpan],
    *,
    document_key: str,
) -> tuple[str, dict[str, str], tuple[_DocumentOffset, ...]]:
    pieces: list[str] = []
    mapping: dict[str, str] = {}
    offsets: list[_DocumentOffset] = []
    source_cursor = 0
    redacted_cursor = 0
    document_digest = document_key.rsplit(":", 1)[-1][:16].upper()

    for index, span in enumerate(spans, start=1):
        prefix = text[source_cursor : span.start]
        pieces.append(prefix)
        redacted_cursor += len(prefix)
        category = _normalize_category(span.label)
        placeholder = f"<<OPENMED_PHI_{category}_{document_digest}_{index:06d}>>"
        pieces.append(placeholder)
        mapping[placeholder] = text[span.start : span.end]
        offsets.append(
            _DocumentOffset(
                placeholder=placeholder,
                redacted_start=redacted_cursor,
                redacted_end=redacted_cursor + len(placeholder),
                source_start=span.start,
                source_end=span.end,
            )
        )
        redacted_cursor += len(placeholder)
        source_cursor = span.end

    pieces.append(text[source_cursor:])
    return "".join(pieces), mapping, tuple(offsets)


def _normalize_category(value: str) -> str:
    category = normalize_label(str(value or OTHER))
    return category if category in CANONICAL_LABELS else OTHER


def _chunk_redacted_document(
    text: str,
    offsets: Sequence[_DocumentOffset],
    *,
    document_key: str,
    chunk_size: int,
) -> tuple[RedactedChunk, ...]:
    if text == "":
        return ()
    chunks: list[RedactedChunk] = []
    start = 0
    ordinal = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        for offset in offsets:
            if offset.redacted_start < end < offset.redacted_end:
                end = offset.redacted_end
                break
        chunk_text = text[start:end]
        chunk_offsets = tuple(
            PlaceholderOffset(
                placeholder=offset.placeholder,
                redacted_start=offset.redacted_start - start,
                redacted_end=offset.redacted_end - start,
                source_start=offset.source_start,
                source_end=offset.source_end,
            )
            for offset in offsets
            if start <= offset.redacted_start and offset.redacted_end <= end
        )
        digest = hashlib.sha256(f"{document_key}:{ordinal}".encode("ascii")).hexdigest()
        chunks.append(
            RedactedChunk(
                chunk_id=f"sha256:{digest}",
                document_key=document_key,
                ordinal=ordinal,
                text=chunk_text,
                placeholder_offsets=chunk_offsets,
            )
        )
        ordinal += 1
        start = end
    return tuple(chunks)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            os.chmod(temporary_path, 0o600)
            json.dump(
                payload,
                handle,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


__all__ = [
    "Deidentifier",
    "INDEX_SCHEMA_VERSION",
    "IndexingResult",
    "PlaceholderOffset",
    "RedactedChunk",
    "RedactedIndex",
]
