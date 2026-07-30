"""Deterministic sharding helpers for distributed batch processing."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_ID_FIELDS = ("id", "document_id", "doc_id")
SHARDING_ALGORITHM = "sha256-id-mod-v1"
HASH_NAMESPACE = "openmed.processing.distributed"

DocumentIdExtractor = Callable[[Any], str]


class ShardingError(ValueError):
    """Base error raised when a document cannot be sharded safely."""


class MissingDocumentIDError(ShardingError):
    """Raised when a document has no usable stable identifier."""


class DuplicateDocumentIDError(ShardingError):
    """Raised when the same stable identifier appears more than once."""


@dataclass(frozen=True)
class DocumentShard:
    """PHI-minimized membership metadata for one deterministic shard."""

    shard_id: int
    document_ids: tuple[str, ...]
    document_hashes: tuple[str, ...]
    fingerprint: str

    @property
    def document_count(self) -> int:
        """Number of documents assigned to this shard."""
        return len(self.document_hashes)

    @property
    def is_empty(self) -> bool:
        """Whether this shard has no assigned documents."""
        return not self.document_hashes

    def to_dict(self, *, include_document_ids: bool = False) -> dict[str, Any]:
        """Return PHI-minimized shard metadata for manifests or reports."""
        data: dict[str, Any] = {
            "shard_id": self.shard_id,
            "document_count": self.document_count,
            "document_hashes": list(self.document_hashes),
            "fingerprint": self.fingerprint,
        }
        if include_document_ids:
            data["document_ids"] = list(self.document_ids)
        return data


@dataclass(frozen=True)
class ShardPlan:
    """Deterministic plan for partitioning documents into stable shards."""

    shard_count: int
    shards: tuple[DocumentShard, ...]
    fingerprint: str
    algorithm: str = SHARDING_ALGORITHM

    @property
    def document_count(self) -> int:
        """Total number of documents covered by the plan."""
        return sum(shard.document_count for shard in self.shards)

    @property
    def non_empty_shards(self) -> tuple[DocumentShard, ...]:
        """Return only shards with assigned documents."""
        return tuple(shard for shard in self.shards if not shard.is_empty)

    def document_to_shard(self) -> dict[str, int]:
        """Map document ids to their assigned shard ids."""
        return {
            document_id: shard.shard_id
            for shard in self.shards
            for document_id in shard.document_ids
        }

    def to_dict(self, *, include_document_ids: bool = False) -> dict[str, Any]:
        """Return PHI-minimized plan metadata for manifests or reports."""
        return {
            "algorithm": self.algorithm,
            "shard_count": self.shard_count,
            "document_count": self.document_count,
            "fingerprint": self.fingerprint,
            "shards": [
                shard.to_dict(include_document_ids=include_document_ids)
                for shard in self.shards
            ],
        }


def stable_document_hash(document_id: str) -> str:
    """Return the stable content-addressed hash used for shard assignment."""
    normalized_id = _normalize_document_id(document_id, index=None)
    material = f"{HASH_NAMESPACE}:{normalized_id}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def assign_document_shard(document_id: str, shard_count: int) -> int:
    """Assign a document id to a stable shard id."""
    _validate_shard_count(shard_count)
    digest = stable_document_hash(document_id)
    return int(digest, 16) % shard_count


def plan_document_shards(
    documents: Iterable[Any],
    *,
    shard_count: int,
    worker_count: int | None = None,
    id_fields: str | Sequence[str] = DEFAULT_ID_FIELDS,
    id_extractor: DocumentIdExtractor | None = None,
) -> ShardPlan:
    """Build a deterministic shard plan from documents with stable ids.

    Shard membership depends only on each document id and ``shard_count``.
    ``worker_count`` is accepted for executor-facing call sites but does not
    affect membership, which lets operators change workers without reshuffling
    an already planned corpus.
    """
    _validate_shard_count(shard_count)
    if worker_count is not None and worker_count < 1:
        raise ValueError("worker_count must be greater than zero")

    normalized_id_fields = _normalize_id_fields(id_fields)
    assignments: dict[int, list[tuple[str, str]]] = defaultdict(list)
    seen_ids: dict[str, int] = {}

    for index, document in enumerate(documents):
        document_id = _extract_document_id(
            document,
            index=index,
            id_fields=normalized_id_fields,
            id_extractor=id_extractor,
        )
        if document_id in seen_ids:
            raise DuplicateDocumentIDError(
                "Duplicate document id encountered at index "
                f"{index}; first seen at index {seen_ids[document_id]}"
            )
        seen_ids[document_id] = index

        document_hash = stable_document_hash(document_id)
        shard_id = int(document_hash, 16) % shard_count
        assignments[shard_id].append((document_hash, document_id))

    shards = tuple(
        _build_document_shard(shard_id, assignments.get(shard_id, []))
        for shard_id in range(shard_count)
    )
    return ShardPlan(
        shard_count=shard_count,
        shards=shards,
        fingerprint=_fingerprint_plan(shards),
    )


def _validate_shard_count(shard_count: int) -> None:
    if shard_count < 1:
        raise ValueError("shard_count must be greater than zero")


def _normalize_id_fields(id_fields: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(id_fields, str):
        fields = (id_fields,)
    else:
        fields = tuple(id_fields)

    if not fields or any(not field for field in fields):
        raise ValueError("id_fields must contain at least one non-empty field name")
    return fields


def _extract_document_id(
    document: Any,
    *,
    index: int,
    id_fields: tuple[str, ...],
    id_extractor: DocumentIdExtractor | None,
) -> str:
    if id_extractor is not None:
        return _normalize_document_id(id_extractor(document), index=index)

    if isinstance(document, Mapping):
        for field in id_fields:
            if field in document:
                return _normalize_document_id(document[field], index=index)
    else:
        for field in id_fields:
            if hasattr(document, field):
                return _normalize_document_id(getattr(document, field), index=index)

    fields = ", ".join(id_fields)
    raise MissingDocumentIDError(
        f"Document at index {index} has no stable id field; expected one of: {fields}"
    )


def _normalize_document_id(document_id: Any, *, index: int | None) -> str:
    if document_id is None:
        if index is None:
            raise MissingDocumentIDError("Document id is required")
        raise MissingDocumentIDError(f"Document at index {index} has no stable id")

    normalized_id = str(document_id).strip()
    if not normalized_id:
        if index is None:
            raise MissingDocumentIDError("Document id must be non-empty")
        raise MissingDocumentIDError(
            f"Document at index {index} has an empty stable id"
        )
    return normalized_id


def _build_document_shard(
    shard_id: int,
    assignments: list[tuple[str, str]],
) -> DocumentShard:
    ordered_assignments = tuple(
        sorted(assignments, key=lambda item: (item[0], item[1]))
    )
    document_hashes = tuple(document_hash for document_hash, _ in ordered_assignments)
    document_ids = tuple(document_id for _, document_id in ordered_assignments)
    return DocumentShard(
        shard_id=shard_id,
        document_ids=document_ids,
        document_hashes=document_hashes,
        fingerprint=_fingerprint_shard(shard_id, document_hashes),
    )


def _fingerprint_shard(shard_id: int, document_hashes: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(f"{HASH_NAMESPACE}:shard:{shard_id}".encode("utf-8"))
    for document_hash in document_hashes:
        digest.update(b"\0")
        digest.update(document_hash.encode("ascii"))
    return digest.hexdigest()


def _fingerprint_plan(shards: tuple[DocumentShard, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(f"{HASH_NAMESPACE}:plan:{len(shards)}".encode("utf-8"))
    for shard in shards:
        digest.update(b"\0")
        digest.update(str(shard.shard_id).encode("ascii"))
        digest.update(b":")
        digest.update(shard.fingerprint.encode("ascii"))
    return digest.hexdigest()
